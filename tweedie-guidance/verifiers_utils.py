import torch
from qwen_vl_utils import process_vision_info
import sys
sys.path.insert(0, "dinov3")
from dinov3.data.transforms import make_classification_eval_transform, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.nn import CosineSimilarity
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn.functional as F
from LVFace.backbones import get_model

class VLMVerifier:
    def __init__(
            self,
            vlm,
            processor,
            question
    ):
        self.vlm = vlm
        self.processor = processor
        self.question = question

    def get_score(self,x0_pred,ref_image):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": ref_image},
                    {"type": "image", "image": x0_pred},
                    {"type": "text", "text": self.question}
                ],
            }
        ]

        # Process the input
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        ).to(self.vlm.device)

        with torch.no_grad():
            outputs = self.vlm(**inputs)
            logits = outputs.logits

        next_token_logits = logits[:, -1, :]

        yes_id = self.processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_id = self.processor.tokenizer.encode("No", add_special_tokens=False)[0]

        # Check dtype and actual values for debugging
        print(f"Logits dtype: {next_token_logits.dtype}")
        print(f"Yes logit (raw): {next_token_logits[0, yes_id].item()}")
        print(f"No logit (raw): {next_token_logits[0, no_id].item()}")

        logit_yes = next_token_logits[0, yes_id].item()
        logit_no = next_token_logits[0, no_id].item()

        # Return raw logit difference for BFS (softmax applied across all particles)
        logit = logit_yes - logit_no

        return logit
    
class DINOVerifier:
    def __init__(
            self,
            dino
    ):
        self.dino = dino

    @torch.no_grad()
    def get_score(self,x0_pred,ref_image):

        pil_transform = make_classification_eval_transform()

        ref_image = pil_transform(ref_image)
        x0_pred = pil_transform(x0_pred)

        ref_feats = self.dino(ref_image.unsqueeze(0).cuda().float())
        gen_feats = self.dino(x0_pred.unsqueeze(0).cuda().float()) 

        sim = CosineSimilarity()
        loss = sim(gen_feats, ref_feats)

        return loss
    
class DINOFaceVerifier:
    def __init__(
            self,
            dino,
            face_app,
            device="cuda"
    ):
        self.dino = dino
        self.face_app = face_app  
        self.device = device
        self.ref_embedding = None         # pre-computed, no grad needed
        self.ref_box = None

    def set_reference(self, ref_image_tensor):  # (1, 3, H, W) in [-1, 1]
        img_np = ((ref_image_tensor[0].permute(1,2,0).cpu().numpy() + 1) * 127.5).astype(np.uint8)[:,:,::-1]
        faces = self.face_app.get(img_np)
        # store bounding box as oracle
        self.ref_box = faces[0].bbox.astype(int)
        with torch.no_grad():
            crop = self._crop(ref_image_tensor, self.ref_box)
            self.ref_embedding = self.dino(crop)

    def _crop(self, x, box):
        x1, y1, x2, y2 = box
        crop = x[:, :, y1:y2, x1:x2]
        crop = F.interpolate(crop.to(self.device), size=224, mode='bicubic')
        # remap [-1,1] → ImageNet-normalized
        crop = (crop + 1) / 2  # → [0,1]
        mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=self.device).view(1,3,1,1)
        std  = torch.tensor(IMAGENET_DEFAULT_STD,  device=self.device).view(1,3,1,1)
        return (crop - mean) / std

    def compute_loss(self, x0_pred):  # x0_pred: (1, 3, H, W) in [-1, 1], WITH grad
        # detect box on current prediction — oracle, no grad
        with torch.no_grad():
            img_np = ((x0_pred[0].detach().permute(1,2,0).cpu().float().numpy() + 1) * 127.5).astype(np.uint8)[:,:,::-1]
            faces = self.face_app.get(img_np)
        
        if not faces:
            return (x0_pred * 0).sum()  # zero but has grad_fn
        
        box = faces[0].bbox.astype(int)
        crop = self._crop(x0_pred, box)          # ← in grad graph
        emb = self.dino(crop)               # ← in grad graph
        loss = torch.abs(emb - self.ref_embedding).mean()
        return loss
    
class IDVerifier:
    def __init__(self, face_app, device='cuda'):
        self.face_app = face_app          # InsightFace — oracle for detection only
        self.device = device
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        self.ref_embedding = None         # pre-computed, no grad needed
        self.ref_box = None

    def set_reference(self, ref_pil):  # PIL image (RGB)
        img_np = np.array(ref_pil)[:, :, ::-1]  # RGB→BGR uint8 for InsightFace
        faces = self.face_app.get(img_np)
        # store bounding box as oracle
        self.ref_box = faces[0].bbox.astype(int)
        # PIL → tensor (1,3,H,W) in [-1,1] for FaceNet
        ref_tensor = (torch.from_numpy(np.array(ref_pil)).permute(2, 0, 1).float() / 127.5 - 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            crop = self._crop(ref_tensor, self.ref_box)
            self.ref_embedding = self.resnet(crop)  # no grad

    def _crop(self, x, box):  # differentiable crop
        x1, y1, x2, y2 = box
        crop = x[:, :, y1:y2, x1:x2]
        return F.interpolate(crop.to(self.device), size=160, mode='bicubic')

    def compute_loss(self, x0_pred):  # x0_pred: (1, 3, H, W) in [-1, 1], WITH grad
        # detect box on current prediction — oracle, no grad
        with torch.no_grad():
            img_np = ((x0_pred[0].detach().permute(1,2,0).cpu().float().numpy() + 1) * 127.5).astype(np.uint8)[:,:,::-1]
            faces = self.face_app.get(img_np)
        
        if not faces:
            return (x0_pred * 0).sum()  # zero but has grad_fn
        
        box = faces[0].bbox.astype(int)
        crop = self._crop(x0_pred, box)          # ← in grad graph
        emb = self.resnet(crop)                  # ← in grad graph
        loss = torch.abs(emb - self.ref_embedding).mean()
        return loss
    
    def get_id_similarity(self,x0_pred,ref_image):
        x0_pred = np.array(x0_pred)[:,:,::-1]
        ref_image = np.array(ref_image)[:,:,::-1]

        faces_pred = self.face_app.get(x0_pred)
        faces_ref = self.face_app.get(ref_image)

        emb_pred = torch.tensor(faces_pred[0].normed_embedding).unsqueeze(0)
        emb_ref = torch.tensor(faces_ref[0].normed_embedding).unsqueeze(0)
        sim = CosineSimilarity()

        return sim(emb_pred,emb_ref)

class LVFaceVerifier:
    def __init__(self, face_app, device='cuda'):
        self.face_app = face_app          # InsightFace — oracle for detection only
        self.device = device
        self.net = get_model("vit_b_dp005_mask_005", fp16=False).to(device)
        self.net.load_state_dict(torch.load("./LVFace/models/lvface/LVFace-B_Glint360K/LVFace-B_Glint360K.pt"))
        self.net.eval()
        self.ref_embedding = None         # pre-computed, no grad needed
        self.ref_box = None

    def set_reference(self, ref_image_tensor):  # (1, 3, H, W) in [-1, 1]
        img_np = ((ref_image_tensor[0].permute(1,2,0).cpu().numpy() + 1) * 127.5).astype(np.uint8)[:,:,::-1]
        faces = self.face_app.get(img_np)
        # store bounding box as oracle
        self.ref_box = faces[0].bbox.astype(int)
        with torch.no_grad():
            crop = self._crop(ref_image_tensor, self.ref_box)
            self.ref_embedding = self.net(crop)  # no grad

    def _crop(self, x, box):  # differentiable crop
        x1, y1, x2, y2 = box
        crop = x[:, :, y1:y2, x1:x2]
        return F.interpolate(crop.to(self.device), size=112, mode='bicubic')

    def compute_loss(self, x0_pred):  # x0_pred: (1, 3, H, W) in [-1, 1], WITH grad
        # detect box on current prediction — oracle, no grad
        with torch.no_grad():
            img_np = ((x0_pred[0].detach().permute(1,2,0).cpu().float().numpy() + 1) * 127.5).astype(np.uint8)[:,:,::-1]
            faces = self.face_app.get(img_np)
        
        if not faces:
            return (x0_pred * 0).sum()  # zero but has grad_fn
        
        box = faces[0].bbox.astype(int)
        crop = self._crop(x0_pred, box)          # ← in grad graph
        emb = self.net(crop)                  # ← in grad graph
        loss = torch.abs(emb - self.ref_embedding).mean()
        return loss
    
    def get_id_similarity(self,x0_pred,ref_image):
        x0_pred = np.array(x0_pred)[:,:,::-1]
        ref_image = np.array(ref_image)[:,:,::-1]

        faces_pred = self.face_app.get(x0_pred)
        faces_ref = self.face_app.get(ref_image)

        emb_pred = torch.tensor(faces_pred[0].normed_embedding).unsqueeze(0)
        emb_ref = torch.tensor(faces_ref[0].normed_embedding).unsqueeze(0)
        sim = CosineSimilarity()

        return sim(emb_pred,emb_ref)
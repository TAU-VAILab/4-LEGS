import torch
from dataclasses import dataclass, field
from typing import Tuple, Type
import torch
import torchvision
from torch import nn
from feature_extraction.ViCLIP.viclip import ViCLIP
from feature_extraction.ViCLIP.simple_tokenizer import SimpleTokenizer as _Tokenizer

BATCH_SIZE = 20

@dataclass
class VICLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: VICLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 768
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)

class VICLIPNetwork(nn.Module):
    def __init__(self, config: VICLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.tokenizer = _Tokenizer()
        model = ViCLIP(self.tokenizer)
        model.eval()
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives    
        self.negatives = self.config.negatives
        with torch.no_grad():                
            text_feat_d = {}
            for phrase in self.positives:
                feat = self.model.get_text_features(phrase, self.tokenizer, text_feat_d)
                text_feat_d[phrase] = feat

            text_feats = [text_feat_d[t] for t in self.positives]
            self.pos_embeds = torch.cat(text_feats, 0)
            text_feat_d = {}
            for phrase in self.negatives:
                feat = self.model.get_text_features(phrase, self.tokenizer, text_feat_d)
                text_feat_d[phrase] = feat

            text_feats = [text_feat_d[t] for t in self.negatives]
            self.neg_embeds = torch.cat(text_feats, 0)
        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "viclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims
    
    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            text_feat_d = {}
            for phrase in self.positives:
                feat = self.model.get_text_features(phrase, self.tokenizer, text_feat_d)
                text_feat_d[phrase] = feat

            text_feats = [text_feat_d[t] for t in self.positives]
            self.pos_embeds = torch.cat(text_feats, 0)

    def set_negatives(self, negs, keep_config_negs=False):
       with torch.no_grad():
            text_feat_d = {}
            if keep_config_negs:
                self.negatives =[neg for neg in self.config.negatives] + negs
            else:
                self.negatives = negs
            for phrase in self.negatives:
                feat = self.model.get_text_features(phrase, self.tokenizer, text_feat_d)
                text_feat_d[phrase] = feat

            text_feats = [text_feat_d[t] for t in self.negatives]
            self.neg_embeds = torch.cat(text_feats, 0)

    def reset_config_negs(self):
        self.negatives = self.config.negatives
        with torch.no_grad():                
            text_feat_d = {}
            for phrase in self.negatives:
                feat = self.model.get_text_features(phrase, self.tokenizer, text_feat_d)
                text_feat_d[phrase] = feat

            text_feats = [text_feat_d[t] for t in self.negatives]
            self.neg_embeds = torch.cat(text_feats, 0)

    def encode_text_latent(self, encoder):
        self.neg_embeds = encoder.encode(self.neg_embeds)
        self.pos_embeds = encoder.encode(self.pos_embeds)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int, negs=None, ret_pos=False, ret_both=False, encoder=None) -> torch.Tensor:
        num_negs = len(self.negatives) if negs is None else len(self.negatives) + len(negs)
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0) 
        p = phrases_embeds.to(embed.dtype) 
        output = torch.mm(embed, p.T) 
        positive_vals = output[..., positive_id : positive_id + 1]  
        if ret_pos:
            return positive_vals
        negative_vals = output[..., len(self.positives) :] 
        repeated_pos = positive_vals.repeat(1, num_negs) 

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  
        softmax = torch.softmax(10 * sims, dim=-1)  
        best_id = softmax[..., 0].argmin(dim=1)  
        if ret_both:
            return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], num_negs, 2))[:, 0, :][...,0:1], positive_vals
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], num_negs, 2))[:, 0, :]
    
    def encode_video(self, input, as_list=False):
        if as_list:
            input_vid = []
            for i,frame in enumerate(input):
                processed_input = self.process(frame)
                input_vid.append(processed_input)
            input_vid = torch.stack(input_vid, dim=0).unsqueeze(0)
        else:
            input_vid = input.reshape(-1,input.shape[2],input.shape[3],input.shape[4])
            input_vid = self.process(input_vid)
            input_vid = input_vid.reshape(input.shape[0], input.shape[1], input_vid.shape[-3], input_vid.shape[-2], input_vid.shape[-1])
        return self.model.get_vid_features(input_vid.cuda())
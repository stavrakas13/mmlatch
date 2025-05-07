import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import numpy as np
import os
import numpy as np
import torch
import umap
import seaborn as sns
import matplotlib.pyplot as plt

from mmlatch.util import plot_umap, truncate_embeddings, fix_nan_inf, compute_metrics_embeddings



from mmlatch.attention import Attention, SymmetricAttention
from mmlatch.rnn import RNN, AttentiveRNN

class FeedbackUnit(nn.Module):
    """Applies the mask to the input. Either a learnable static mask or a learnable sequence mask"""
    def __init__(
        self,
        hidden_dim,
        mod1_sz, # size of the first modality(mask size)
        mask_type="learnable_sequence_mask",
        dropout=0.1,
        device="cpu",
        mask_index = 1, #add mask index parameter
    ):
        super(FeedbackUnit, self).__init__()
        self.mask_type = mask_type
        self.mod1_sz = mod1_sz
        self.hidden_dim = hidden_dim
        self.mask_index = mask_index

        if mask_type == "learnable_sequence_mask":#if the mask is learnable sequence mask, initialize two RNNs
            self.mask1 = RNN(hidden_dim, mod1_sz, dropout=dropout, device=device)
            self.mask2 = RNN(hidden_dim, mod1_sz, dropout=dropout, device=device)
        else: #if the mask is learnable static mask, initialize two linear layers
            self.mask1 = nn.Linear(hidden_dim, mod1_sz)
            self.mask2 = nn.Linear(hidden_dim, mod1_sz)

        mask_fn = {#dictionary to store the mask functions
            "learnable_static_mask": self._learnable_static_mask,
            "learnable_sequence_mask": self._learnable_sequence_mask,
        }

        self.get_mask = mask_fn[self.mask_type] #get the mask function based on the mask type

    def _learnable_sequence_mask(self, y, z, lengths=None,used_y = True,used_z = True):
        """EDITS HERE !!!!!"""
        """Generates a dynamic, per-timestep mask based on the feedback from inputs y and z"""
        """Mask index:
        1-> average
        2-> element_wise maximum
        3-> element_wise minimum
        4-> residual connection(add original again)
        5-> element wise max deviation from 0.5
        """
        oy, _, _ = self.mask1(y, lengths)
        oz, _, _ = self.mask2(z, lengths)
        mask1 = torch.sigmoid(oy)
        mask2 = torch.sigmoid(oz)
        if(used_y == False):
            mask =mask2
        elif(used_z == False):
            mask = mask1
        elif self.mask_index == 1 or self.mask_index == 4 or self.mask_index == 6:
            mask = (mask1 + mask2) * 0.5
        elif self.mask_index == 2:
            mask = torch.max(mask1, mask2)
        elif self.mask_index == 3:
            mask = torch.min(mask1, mask2)
        elif self.mask_index == 5:
            mask = torch.where(torch.abs(mask1 - 0.5) > torch.abs(mask2 - 0.5), mask1, mask2)
        else:
            raise ValueError("Invalid mask_index. Must be between 1 and 5.")

        
        #lg = (torch.sigmoid(oy) + torch.sigmoid(oz)) * 0.5

        #mask = lg

        return mask,mask1

    #def _learnable_static_mask(self, y, z, lengths=None):
    def _learnable_static_mask(self, y, z, lengths=None,used_y = True,used_z = True):
        """Generates a static mask based on the feedback from inputs y and z"""
        y = self.mask1(y)
        z = self.mask2(z)
        mask1 = torch.sigmoid(y)
        mask2 = torch.sigmoid(z)
        #mask = (mask1 + mask2) * 0.5
        if(used_y == False):
            mask =mask2
        elif(used_z == False):
            mask = mask1
        elif self.mask_index == 1 or self.mask_index == 4:
            mask = (mask1 + mask2) * 0.5
        elif self.mask_index == 2:
            mask = torch.max(mask1, mask2)
        elif self.mask_index == 3:
            mask = torch.min(mask1, mask2)
        elif self.mask_index == 4:
            mask = (mask1 + mask2) * 0.5 + 1  
        elif self.mask_index == 5:
            mask = torch.where(torch.abs(mask1 - 0.5) > torch.abs(mask2 - 0.5), mask1, mask2)

        else:
            raise ValueError("Invalid mask_index. Must be between 1 and 5.")

        return mask,mask1
    def set_mask_index(self, new_mask_index):
        """
        Updates the mask_index to a new value.
        
        Args:
            new_mask_index (int): New mask index value (1 to 5).
        """
        if new_mask_index not in [1, 2, 3, 4, 5, 6]:
            raise ValueError("mask_index must be an integer between 1 and 5.")
        self.mask_index = new_mask_index

    def forward(self, x, y, z, lengths=None,used_y = True,used_z = True):
        """Applies the mask to the modality x based on the feedback from y and z"""
        mask,mask1 = self.get_mask(y, z, lengths=lengths,used_y=used_y,used_z=used_z)
        mask = F.dropout(mask, p=0.2) #apply dropout to the mask
        if(self.mask_index == 4): # add the residual part after dropout
            x_new = x/3 + x*mask
        if(self.mask_index == 6):
            x_new = x
        else:
            x_new = x * mask
        
        
        #return x_new,mask1
        return x_new,mask


class Feedback(nn.Module):
    """Feedback mechanism to apply feedback-based masking across three modalities (x, y, z"""
    def __init__(
        self,
        hidden_dim,
        mod1_sz,
        mod2_sz,
        mod3_sz,
        mask_type="learnable_sequence_mask",
        dropout=0.1,
        device="cpu",
        mask_index = 1, #add mask index parameter
        mask_dropout = 0.0, #add mask dropout parameter
    ):
        super(Feedback, self).__init__()
        # Initialize a feedback unit for each modality
        self.mask_dropout = mask_dropout
        self.f1 = FeedbackUnit(
            hidden_dim,
            mod1_sz,
            mask_type=mask_type,
            dropout=dropout,
            device=device,
            mask_index=mask_index
        )
        self.f2 = FeedbackUnit(
            hidden_dim,
            mod2_sz,
            mask_type=mask_type,
            dropout=dropout,
            device=device,
            mask_index=mask_index
        )
        self.f3 = FeedbackUnit(
            hidden_dim,
            mod3_sz,
            mask_type=mask_type,
            dropout=dropout,
            device=device,
            mask_index=mask_index
        )
    def set_mask_index(self, new_mask_index):
        """
        Updates the mask_index for all FeedbackUnit instances.
        
        Args:
            new_mask_index (int): New mask index value (1 to 5).
        """
        self.f1.set_mask_index(new_mask_index)
        self.f2.set_mask_index(new_mask_index)
        self.f3.set_mask_index(new_mask_index)
    def set_mask_dropout(self, new_mask_dropout):
        """Updates mask_dropout for all Feedback ."""
        self.mask_dropout = new_mask_dropout

    def forward(self, low_x, low_y, low_z, hi_x, hi_y, hi_z, lengths=None):
        """Applies the feedback mechanism across three modalities"""
        used =[True,True,True]

        if self.training:
            # 20% chance to drop one of the three masks
            if torch.rand(1).item() < self.mask_dropout:
                drop_idx = torch.randint(0, 3, (1,)).item()
                used[drop_idx] = False


        #x,masky = self.f1(low_x, hi_y, hi_z, lengths=lengths,used_y = used[1],used_z = used[2])
        #y,maskz = self.f2(low_y, hi_z, hi_x, lengths=lengths,used_y = used[2],used_z = used[0])
        #z,maskx = self.f3(low_z, hi_x, hi_y, lengths=lengths,used_y = used[0],used_z = used[1])

        x,mask_to_x = self.f1(low_x, hi_y, hi_z, lengths=lengths,used_y = used[1],used_z = used[2])
        y,mask_to_y= self.f2(low_y, hi_z, hi_x, lengths=lengths,used_y = used[2],used_z = used[0])
        z,mask_to_z = self.f3(low_z, hi_x, hi_y, lengths=lengths,used_y = used[0],used_z = used[1])

        #return x, y, z,maskx,masky,maskz
        return x, y, z,mask_to_x,mask_to_y,mask_to_z


class AttentionFuser(nn.Module):
    """ combines multiple attention mechanisms to fuse information from three modalities"""
    def __init__(self, proj_sz=None, return_hidden=True, device="cpu"):
        super(AttentionFuser, self).__init__()
        self.return_hidden = return_hidden
        #Text-Audio attention
        self.ta = SymmetricAttention(
            attention_size=proj_sz,
            dropout=0.1,
        )
        #Video-Audio attention
        self.va = SymmetricAttention(
            attention_size=proj_sz,
            dropout=0.1,
        )
        #Text-Video attention
        self.tv = SymmetricAttention(
            attention_size=proj_sz,
            dropout=0.1,
        )
        #Text-Audio-Video attention
        self.tav = Attention(
            attention_size=proj_sz,
            dropout=0.1,
        )
        #output size of the fuser
        self.out_size = 7 * proj_sz

    def forward(self, txt, au, vi):
        # Apply cross modal attention mechanisms
        ta, at = self.ta(txt, au) 
        va, av = self.va(vi, au)
        tv, vt = self.tv(txt, vi)
        #combine the attention outputs
        av = va + av
        tv = vt + tv
        ta = ta + at

        tav, _ = self.tav(txt, queries=av) 

        # Sum weighted attention hidden states

        if not self.return_hidden:
            txt = txt.sum(1)
            au = au.sum(1)
            vi = vi.sum(1)
            ta = ta.sum(1)
            av = av.sum(1)
            tv = tv.sum(1)
            tav = tav.sum(1)

        # B x L x 7*D
        #concatenate the attention outputs
        fused = torch.cat([txt, au, vi, ta, tv, av, tav], dim=-1)

        return fused


class AttRnnFuser(nn.Module):
    """ AttRnnFuser class integrates the AttentionFuser with an AttentiveRNN to further process the fused representations"""
    def __init__(self, proj_sz=None, device="cpu", return_hidden=False):
        super(AttRnnFuser, self).__init__()
        self.att_fuser = AttentionFuser( #initialize the attention fuser
            proj_sz=proj_sz,
            return_hidden=True,
            device=device,
        )
        self.rnn = AttentiveRNN( #initialize the AttentiveRNN
            self.att_fuser.out_size,
            proj_sz,
            bidirectional=True,
            merge_bi="cat",
            attention=True,
            device=device,
            return_hidden=return_hidden,
        )
        self.out_size = self.rnn.out_size

    def forward(self, txt, au, vi, lengths):
        """Forward pass of the AttRnnFuser"""
        att = self.att_fuser(txt, au, vi)  # B x L x 7 * D
        out = self.rnn(att, lengths)  # B x L x 2 * D

        return out


class AudioEncoder(nn.Module):
    """AttentiveRNN tailored for processing audio data"""
    def __init__(self, cfg, device="cpu"):
        super(AudioEncoder, self).__init__()
        self.audio = AttentiveRNN(
            cfg["input_size"],
            cfg["hidden_size"],
            batch_first=True,
            layers=cfg["layers"],
            merge_bi="sum",
            bidirectional=cfg["bidirectional"],
            dropout=cfg["dropout"],
            rnn_type=cfg["rnn_type"],
            packed_sequence=True,
            attention=cfg["attention"],
            device=device,
            return_hidden=cfg["return_hidden"],
        )
        self.out_size = self.audio.out_size

    def forward(self, x, lengths):
        x = self.audio(x, lengths)

        return x


class VisualEncoder(nn.Module):
    """AttentiveRNN tailored for processing visual data"""
    def __init__(self, cfg, device="cpu"):
        super(VisualEncoder, self).__init__()
        self.visual = AttentiveRNN(
            cfg["input_size"],
            cfg["hidden_size"],
            batch_first=True,
            layers=cfg["layers"],
            merge_bi="sum",
            bidirectional=cfg["bidirectional"],
            dropout=cfg["dropout"],
            rnn_type=cfg["rnn_type"],
            packed_sequence=True,
            attention=cfg["attention"],
            device=device,
            return_hidden=cfg["return_hidden"],
        )
        self.out_size = self.visual.out_size

    def forward(self, x, lengths):
        x = self.visual(x, lengths)

        return x


class GloveEncoder(nn.Module):
    """AttentiveRNN tailored for processing text data"""
    def __init__(self, cfg, device="cpu"):
        super(GloveEncoder, self).__init__()
        self.text = AttentiveRNN(
            cfg["input_size"],
            cfg["hidden_size"],
            batch_first=True,
            layers=cfg["layers"],
            merge_bi="sum",
            bidirectional=cfg["bidirectional"],
            dropout=cfg["dropout"],
            rnn_type=cfg["rnn_type"],
            packed_sequence=True,
            attention=cfg["attention"],
            device=device,
            return_hidden=cfg["return_hidden"],
        )
        self.out_size = self.text.out_size

    def forward(self, x, lengths):
        x = self.text(x, lengths)

        return x


class AudioVisualTextEncoder(nn.Module):
    """AudioVisualTextEncoder class integrates the encoders for text, audio, and visual modalities"""
    def __init__(
        self,
        audio_cfg=None,
        text_cfg=None,
        visual_cfg=None,
        fuse_cfg=None,
        feedback=False,
        device="cpu",
    ):
        super(AudioVisualTextEncoder, self).__init__()
        assert (
            text_cfg["attention"] and audio_cfg["attention"] and visual_cfg["attention"]
        ), "Use attention pls."

        self.feedback = feedback
        text_cfg["return_hidden"] = True
        audio_cfg["return_hidden"] = True
        visual_cfg["return_hidden"] = True

        self.text = GloveEncoder(text_cfg, device=device)
        #self.text = BertEncoder(cfg, device="cuda")
        self.audio = AudioEncoder(audio_cfg, device=device)
        self.visual = VisualEncoder(visual_cfg, device=device)

        self.fuser = AttRnnFuser(
            proj_sz=fuse_cfg["projection_size"],
            device=device,
        )

        self.out_size = self.fuser.out_size

        if feedback:
            self.fm = Feedback(
                fuse_cfg["projection_size"],
                text_cfg["input_size"],
                audio_cfg["input_size"],
                visual_cfg["input_size"],
                mask_type=fuse_cfg["feedback_type"],
                dropout=0.1,
                device=device,
                mask_index=fuse_cfg.get("mask_index", 1),
            )

    def _encode(self, txt, au, vi, lengths):
        """Encodes the input data for each modality"""
        txt = self.text(txt, lengths)
        au = self.audio(au, lengths)
        vi = self.visual(vi, lengths)

        return txt, au, vi

    def _fuse(self, txt, au, vi, lengths):
        """Fuses the encoded representations from the three modalities for classifier input"""
        fused = self.fuser(txt, au, vi, lengths)

        return fused
    """
    def forward(self, txt, au, vi, lengths):
        if self.feedback:
            txt1, au1, vi1 = self._encode(txt, au, vi, lengths)
            txt, au, vi, mask_to_x,mask_to_y,mask_to_z = self.fm(txt, au, vi, txt1, au1, vi1, lengths=lengths)

        txt, au, vi = self._encode(txt, au, vi, lengths)
        fused = self._fuse(txt, au, vi, lengths)

        return fused
    """

    def forward(self, txt, au, vi, lengths):
        if self.feedback:
            txt1, au1, vi1 = self._encode(txt, au, vi, lengths)
            txt, au, vi,mask_txt,mask_au,mask_vi = self.fm(txt, au, vi, txt1, au1, vi1, lengths=lengths)

        txt, au, vi = self._encode(txt, au, vi, lengths)
        fused = self._fuse(txt, au, vi, lengths)

        return fused,mask_txt,mask_au,mask_vi

class AudioVisualTextClassifier(nn.Module):
    """AudioVisualTextClassifier class integrates the AudioVisualTextEncoder with a classifier"""
    def __init__(
        self,
        audio_cfg=None,
        text_cfg=None,
        visual_cfg=None,
        fuse_cfg=None,
        modalities=None,
        num_classes=1,
        feedback=False,
        device="cpu",
    ):
        super(AudioVisualTextClassifier, self).__init__()
        self.modalities = modalities

        assert "text" in modalities, "No text"
        assert "audio" in modalities, "No audio"
        assert "visual" in modalities, "No visual"

        self.encoder = AudioVisualTextEncoder(
            text_cfg=text_cfg,
            audio_cfg=audio_cfg,
            visual_cfg=visual_cfg,
            fuse_cfg=fuse_cfg,
            feedback=feedback,
            device=device,
        )

        self.classifier = nn.Linear(self.encoder.out_size, num_classes)

    def forward(self, inputs):
        out, mask_txt,mask_au,mask_vi = self.encoder(
            inputs["text"], inputs["audio"], inputs["visual"], inputs["lengths"]
        )

        return self.classifier(out), mask_txt,mask_au,mask_vi


class UnimodalEncoder(nn.Module):
    """ a generalized encoder for a single modality, encapsulating an AttentiveRNN"""
    def __init__(
        self,
        input_size,
        projection_size,
        layers=1,
        bidirectional=True,
        dropout=0.2,
        encoder_type="lstm",
        attention=True,
        return_hidden=False,
        device="cpu",
    ):
        super(UnimodalEncoder, self).__init__()
        self.encoder = AttentiveRNN(
            input_size,
            projection_size,
            batch_first=True,
            layers=layers,
            merge_bi="sum",
            bidirectional=bidirectional,
            dropout=dropout,
            rnn_type=encoder_type,
            packed_sequence=True,
            attention=attention,
            device=device,
            return_hidden=return_hidden,
        )
        self.out_size = self.encoder.out_size

    def forward(self, x, lengths):
        return self.encoder(x, lengths)


class AVTEncoder(nn.Module):
    """A variant of AudioVisualText Encoder, uses Unimodal Encoder"""
    def __init__(
        self,
        text_input_size,
        audio_input_size,
        visual_input_size,
        projection_size,
        text_layers=1,
        audio_layers=1,
        visual_layers=1,
        bidirectional=True,
        dropout=0.2,
        encoder_type="lstm",
        attention=True,
        feedback=False,
        feedback_type="learnable_sequence_mask",
        device="cpu",
        mask_index=1,  # Add mask_index parameter
        mask_dropout=0.0,  # Add mask_dropout parameter
    ):
        self.mask_index = mask_index
        super(AVTEncoder, self).__init__()
        self.feedback = feedback

        self.all_txt_embeddings_before = []
        self.all_au_embeddings_before = []
        self.all_vi_embeddings_before = []

        self.all_txt_embeddings_after = []
        self.all_au_embeddings_after = []
        self.all_vi_embeddings_after = []

        self.fused_before = []
        self.fused_after = []

        self.text = UnimodalEncoder(
            text_input_size,
            projection_size,
            layers=text_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            encoder_type=encoder_type,
            attention=attention,
            return_hidden=True,
            device=device,
        )

        self.audio = UnimodalEncoder(
            audio_input_size,
            projection_size,
            layers=audio_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            encoder_type=encoder_type,
            attention=attention,
            return_hidden=True,
            device=device,
        )

        self.visual = UnimodalEncoder(
            visual_input_size,
            projection_size,
            layers=visual_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            encoder_type=encoder_type,
            attention=attention,
            return_hidden=True,
            device=device,
        )

        self.fuser = AttRnnFuser(
            proj_sz=projection_size,
            device=device,
        )
        self.batch_size = 0
        self.out_size = self.fuser.out_size

        if feedback:
            self.fm = Feedback(
                projection_size,
                text_input_size,
                audio_input_size,
                visual_input_size,
                mask_type=feedback_type,
                dropout=0.1,
                device=device,
                mask_index=mask_index,  # Pass mask_index
                mask_dropout=mask_dropout,  # Pass mask_dropout
            )

    def _encode(self, txt, au, vi, lengths):
        txt = self.text(txt, lengths)
        au = self.audio(au, lengths)
        vi = self.visual(vi, lengths)

        return txt, au, vi
    
    def set_mask_index(self, new_mask_index):
        """
        Updates the mask_index for all FeedbackUnit instances.
        
        Args:
            new_mask_index (int): New mask index value (1 to 5).
        """
        self.fm.set_mask_index(new_mask_index)
    def set_mask_dropout(self, new_mask_dropout):
        """Updates mask_dropout for all Feedback ."""
        self.fm.set_mask_dropout(new_mask_dropout)
        

    def _fuse(self, txt, au, vi, lengths):
        fused = self.fuser(txt, au, vi, lengths)

        return fused

    def forward(self, txt, au, vi, lengths, enable_plot_embeddings):
        
        if self.feedback:
            txt1, au1, vi1 = self._encode(txt, au, vi, lengths)

            if enable_plot_embeddings:
                # Apply mean pooling along the sequence dimension (dim=1)
                self.all_txt_embeddings_before.append(txt1.mean(dim=1).detach().cpu().numpy())
                self.all_au_embeddings_before.append(au1.mean(dim=1).detach().cpu().numpy())
                self.all_vi_embeddings_before.append(vi1.mean(dim=1).detach().cpu().numpy())

                fused_before = self._fuse(txt1, au1, vi1, lengths)
                self.fused_before.append(fused_before.detach().cpu().numpy())

            txt, au, vi, mask_txt, mask_au, mask_vi = self.fm(txt, au, vi, txt1, au1, vi1, lengths=lengths)

        txt, au, vi = self._encode(txt, au, vi, lengths)
        fused = self._fuse(txt, au, vi, lengths)

        if enable_plot_embeddings:
            self.all_txt_embeddings_after.append(txt.mean(dim=1).detach().cpu().numpy())
            self.all_au_embeddings_after.append(au.mean(dim=1).detach().cpu().numpy())
            self.all_vi_embeddings_after.append(vi.mean(dim=1).detach().cpu().numpy())
            self.fused_after.append(fused.detach().cpu().numpy())

        return fused, mask_txt, mask_au, mask_vi, txt, au, vi

    
    

    def plot_embeddings(self, targets, results_dir):
        if not self.all_txt_embeddings_before:
            print("No embeddings stored for plotting.")
            return

        modalities = ['txt', 'au', 'vi']
        fig, axes = plt.subplots(4, 2, figsize=(12, 24))  # Added an extra row for fused embeddings
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
        save_dir = f"{results_dir}/plot-representations"
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/embeddings_mask_{self.mask_index}_{timestamp}.png"

        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()
        
        targets = np.clip(np.round(targets).astype(int), -3, 3)

        for i, modality in enumerate(modalities):
            before_list = getattr(self, f"all_{modality}_embeddings_before")
            after_list = getattr(self, f"all_{modality}_embeddings_after")

            before_list = truncate_embeddings(before_list)
            after_list = truncate_embeddings(after_list)

            before_2d = np.concatenate(before_list, axis=0)
            after_2d = np.concatenate(after_list, axis=0)

            before_2d = fix_nan_inf(before_2d)
            after_2d = fix_nan_inf(after_2d)

            embeddings_before = {"before": before_2d, "after": after_2d}

            silhouette_before, cos_sim_before, dbi_before = compute_metrics_embeddings(embeddings_before["before"], targets)
            silhouette_after, cos_sim_after, dbi_after = compute_metrics_embeddings(embeddings_before["after"], targets)

            print(f"{modality} Silhouette score before: {silhouette_before}")
            print(f"{modality} Silhouette score after: {silhouette_after}")
            print(f"{modality} Cosine similarity before: {cos_sim_before}")
            print(f"{modality} Cosine similarity after: {cos_sim_after}")
            print(f"{modality} Davies-Bouldin Index before: {dbi_before}")
            print(f"{modality} Davies-Bouldin Index after: {dbi_after}")

            plot_umap(
                embeddings_before, 
                targets,
                f"{modality.upper()} Embeddings Before\nSilhouette: {silhouette_before:.2f}, Cosine Sim: {cos_sim_before:.2f}, DBI: {dbi_before:.2f}",
                f"{modality.upper()} Embeddings After\nSilhouette: {silhouette_after:.2f}, Cosine Sim: {cos_sim_after:.2f}, DBI: {dbi_after:.2f}",
                axes[i, 0], axes[i, 1], save_path
            )

        # Separate plot for fused embeddings
        save_fused_path = f"{save_dir}/fused_embeddings_mask_{self.mask_index}_{timestamp}.png"

        fused_before_2d = np.concatenate(self.fused_before, axis=0)
        fused_after_2d = np.concatenate(self.fused_after, axis=0)

        #fused_before_2d = fix_nan_inf(fused_before_2d)
        #fused_after_2d = fix_nan_inf(fused_after_2d)

        fused_embeddings = {"before": fused_before_2d, "after": fused_after_2d}
        
        silhouette_before, cos_sim_before, dbi_before = compute_metrics_embeddings(fused_embeddings["before"], targets)
        silhouette_after, cos_sim_after, dbi_after = compute_metrics_embeddings(fused_embeddings["after"], targets)
        
        print(f"Fused Silhouette score before: {silhouette_before}")
        print(f"Fused Silhouette score after: {silhouette_after}")
        print(f"Fused Cosine similarity before: {cos_sim_before}")
        print(f"Fused Cosine similarity after: {cos_sim_after}")
        print(f"Fused Davies-Bouldin Index before: {dbi_before}")
        print(f"Fused Davies-Bouldin Index after: {dbi_after}")
        
        plot_umap(
            fused_embeddings, 
            targets,
            f"Fused Embeddings Before\nSilhouette: {silhouette_before:.2f}, Cosine Sim: {cos_sim_before:.2f}, DBI: {dbi_before:.2f}",
            f"Fused Embeddings After\nSilhouette: {silhouette_after:.2f}, Cosine Sim: {cos_sim_after:.2f}, DBI: {dbi_after:.2f}",
            axes[3, 0], axes[3, 1], save_path
        )

        plt.tight_layout()
        plt.savefig(save_fused_path)
        plt.show()






class AVTClassifier(nn.Module):
    """AVTEncoder + Final Classifier"""
    def __init__(
        self,
        text_input_size,
        audio_input_size,
        visual_input_size,
        projection_size,
        text_layers=1,
        audio_layers=1,
        visual_layers=1,
        bidirectional=True,
        dropout=0.2,
        encoder_type="lstm",
        attention=True,
        feedback=False,
        feedback_type="learnable_sequence_mask",
        device="cpu",
        num_classes=1,
        mask_index=1,  # Add mask_index parameter
        mask_dropout=0.0,  # Add mask_dropout parameter
        enable_plot_embeddings = False
    ):
        self.enable_plot_embeddings = enable_plot_embeddings
        super(AVTClassifier, self).__init__()

        self.encoder = AVTEncoder(
            text_input_size,
            audio_input_size,
            visual_input_size,
            projection_size,
            text_layers=text_layers,
            audio_layers=audio_layers,
            visual_layers=visual_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            encoder_type=encoder_type,
            attention=attention,
            feedback=feedback,
            feedback_type=feedback_type,
            device=device,
            mask_index=mask_index,  # Pass mask_index
            mask_dropout=mask_dropout,  # Pass mask_dropout
        )

        self.classifier = nn.Linear(self.encoder.out_size, num_classes)

    def set_mask_index(self, new_mask_index):
        """
        Updates the mask_index for all FeedbackUnit instances.
        
        Args:
            new_mask_index (int): New mask index value (1 to 5).
        """
        self.encoder.set_mask_index(new_mask_index)
    def set_mask_dropout(self, new_mask_dropout):
        """Updates mask_dropout for all Feedback ."""
        self.encoder.set_mask_dropout(new_mask_dropout)
    def forward(self, inputs, enable_plot_embeddings):
        out,mask_txt,mask_au,mask_vi, txt, au, vi = self.encoder(
            inputs["text"], inputs["audio"], inputs["visual"], inputs["lengths"], enable_plot_embeddings
        )

        return self.classifier(out),mask_txt,mask_au,mask_vi, txt, au, vi
    
    def plot_embeddings(self, targets=None, results_dir=None):
        self.encoder.plot_embeddings(targets, results_dir)

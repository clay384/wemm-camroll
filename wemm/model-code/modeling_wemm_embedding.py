import torch
import torch.nn.functional as F
from transformers import Qwen3_5ForConditionalGeneration


class WeMMEmbedding(Qwen3_5ForConditionalGeneration):
    def embedding(self, input_ids=None, attention_mask=None, **kwargs):
        # transformers < 5.15 reuses the rope_deltas cached by the previous multimodal
        # forward for a text-only one, which shifts its position ids.
        self.model.rope_deltas = None

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        last_hidden_state = outputs.last_hidden_state

        if attention_mask is not None:
            eos_positions = attention_mask.sum(dim=1) - 1
        else:
            eos_positions = torch.full((last_hidden_state.shape[0],), last_hidden_state.shape[1] - 1, device=last_hidden_state.device)

        eos_positions = eos_positions.clamp(min=0)
        
        batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
        
        embeddings = last_hidden_state[batch_indices, eos_positions]
        
        embeddings = F.normalize(embeddings, dim=-1)
        
        return embeddings
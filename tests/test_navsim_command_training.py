import torch

from bevlane.train import evaluate_ego


class _IntentProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_intent = None

    def forward(self, imgs, intrinsics, transforms, speed, intent=None):
        self.seen_intent = None if intent is None else intent.detach().cpu().clone()
        batch = imgs.shape[0]
        outputs = [torch.zeros(batch, 1, 1, 1) for _ in range(7)]
        outputs.append(torch.zeros(batch, 15))
        return tuple(outputs)


def test_evaluate_ego_passes_raw_command_to_model():
    model = _IntentProbe()
    ego = torch.zeros(1, 17)
    ego[:, 12] = 3.0
    ego[:, 16] = 1.0
    command = torch.tensor([[0.0, 1.0, 0.0]])
    batch = (
        torch.zeros(1, 8, 3, 4, 4),
        torch.zeros(1, 8, 3, 3),
        torch.zeros(1, 8, 4, 4),
        torch.zeros(1, 2, 2, dtype=torch.long),
        ego,
        command,
    )

    result = evaluate_ego(
        model, [batch], torch.device("cpu"), ego_idx=4,
        max_batches=1, command_idx=-1,
    )

    assert result is not None
    torch.testing.assert_close(model.seen_intent, command)

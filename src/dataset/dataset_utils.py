import torch


def stop_to_mask(stop_targets: torch.Tensor, max_len: int) -> torch.Tensor:
    batch_size = stop_targets.size(0)

    range_tensor = (
        torch.arange(max_len, device=stop_targets.device)
        .unsqueeze(0)
        .repeat(batch_size, 1)
    )

    expanded_stops = stop_targets.repeat(1, max_len).to(torch.int64)
    mask = (range_tensor < expanded_stops).to(torch.float32)

    return mask


def pad_to_max_len(polys, max_len, padding_value=0):
    """
    Pads a list of tensors to a given maximum length with a specified padding value.

    Args:
        polys (list of torch.Tensor): List of 2D tensors representing polygons. Each tensor is of shape [num_points, 3].
        max_len (int): The desired maximum length (number of points) for the tensors.
        padding_value (int): The value used to pad the tensors.

    Returns:
        torch.Tensor: A padded 3D tensor of shape [batch_size, max_len, 3] where batch_size is the length of the list `polys`.
    """
    batch_size = len(polys)
    # Create an empty tensor for the padded polygons with the desired shape and padding value
    padded_polys = torch.full(
        (batch_size, max_len, 3),
        padding_value,
        dtype=polys[0].dtype,
        device=polys[0].device,
    )

    # Fill the padded tensor with the polygons' points
    for i, poly in enumerate(polys):
        num_points = poly.size(0)
        if num_points:
            padded_polys[i, :num_points, :] = poly

    return padded_polys


def _test_pad_to_max_len():
    polys = [
        torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),
        torch.tensor([[7, 8, 9]], dtype=torch.float32),
    ]

    padded_polys = pad_to_max_len(polys, max_len=3)

    torch.testing.assert_close(
        padded_polys,
        torch.tensor(
            [
                [[1, 2, 3], [4, 5, 6], [0, 0, 0]],
                [[7, 8, 9], [0, 0, 0], [0, 0, 0]],
            ],
            dtype=torch.float32,
        ),
    )


if __name__ == "__main__":
    _test_pad_to_max_len()

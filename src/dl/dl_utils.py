import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

resnet_transform = transforms.Compose(
    [
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Lambda(lambda x: x / 255.0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

vit_transform = transforms.Compose(
    [
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Lambda(lambda x: x / 255.0),
    ]
)


def pts_transform(pts):
    return pts / 100.0


def split_dataset(
    dataset: torch.utils.data.Dataset,
    split_ratio: tuple = (0.8, 0.1, 0.1),
    seed: int = 42,
):
    torch.manual_seed(seed)
    n = len(dataset)
    train_len = int(n * split_ratio[0])
    val_len = int(n * split_ratio[1])
    test_len = n - train_len - val_len
    return torch.utils.data.random_split(dataset, [train_len, val_len, test_len])


def pad_sequences(batch: list, target_len: int, padding_value: int = 0) -> torch.Tensor:
    padded_batch = []
    for tensor in batch:
        padded_tensor = F.pad(
            tensor, (0, 0, 0, target_len - tensor.size(0)), value=padding_value
        )
        padded_batch.append(padded_tensor)
    return torch.stack(padded_batch, dim=0)


def make_masks(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    range_tensor = (
        torch.arange(max_len)
        .unsqueeze(0)
        .expand(lengths.size(0), -1)
        .to(lengths.device)
    )

    expanded_lengths = lengths.unsqueeze(1).expand(-1, max_len).to(lengths.device)

    # Create the mask by comparing each element in range_tensor with expanded_lengths
    mask = range_tensor < expanded_lengths

    return mask.float()


def cartesian_to_spherical(pts):
    x, y, z = pts.unbind(dim=-1)
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.acos(z / r)
    phi = torch.atan2(y, x)
    return torch.stack((theta, phi), dim=-1)


def spherical_to_cartesian(pts, radius):
    theta, phi = pts.unbind(dim=-1)
    x = radius * torch.sin(theta) * torch.cos(phi)
    y = radius * torch.sin(theta) * torch.sin(phi)
    z = radius * torch.cos(theta)
    return torch.stack((x, y, z), dim=-1)


def spherical_offsets_to_cartesian_chain(
    initial_point: torch.Tensor,
    offsets: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    chain = [initial_point]

    for theta, phi in offsets:
        # Convert the angles' tensors to the same device as the initial point
        theta, phi = theta.to(initial_point.device), phi.to(initial_point.device)

        # Calculate dx, dy, dz
        dx = radius * torch.sin(theta) * torch.cos(phi)
        dy = radius * torch.sin(theta) * torch.sin(phi)
        dz = radius * torch.cos(theta)

        # Create a new point tensor directly on the correct device and with the correct dtype
        new_point = chain[-1] + torch.tensor(
            [dx, dy, dz], device=initial_point.device, dtype=initial_point.dtype
        )
        chain.append(new_point)

    # Stack all points into a single tensor to form the chain
    cartesian_chain = torch.stack(chain)

    return cartesian_chain


def cartesian_to_spherical_chain(cartesian_points: torch.Tensor) -> torch.Tensor:
    spherical_coords = []

    # Calculate vector differences and convert to spherical coordinates
    for i in range(1, len(cartesian_points)):
        delta = cartesian_points[i] - cartesian_points[i - 1]
        x, y, z = delta[0], delta[1], delta[2]
        r = torch.sqrt(x**2 + y**2 + z**2)

        # Compute theta and phi
        if r == 0:  # Avoid division by zero
            theta = torch.tensor(0.0)
            phi = torch.tensor(0.0)
        else:
            theta = torch.acos(z / r)
            phi = torch.atan2(y, x)

        spherical_coords.append(torch.tensor([theta, phi]))

    return torch.stack(spherical_coords)


def make_stop_targets(lengths: torch.Tensor, max_len: int):
    stop_targets = torch.zeros(lengths.size(0), max_len).to(lengths.device)
    for i, length in enumerate(lengths):
        if length > 0:
            stop_targets[i, length - 1] = 1.0
    return stop_targets


if __name__ == "__main__":
    random_pts = torch.rand(5, 3)
    spherical_pts = cartesian_to_spherical(random_pts)
    print(spherical_pts.shape)
    print(spherical_pts)

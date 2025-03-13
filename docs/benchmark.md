#  Grounding-PanopticSports Benchmark

Download the benchmark from [here](https://drive.google.com/drive/folders/1iZ1HvEUz-xQtWJY2aoXIcPhX_owVVj6b?usp=drive_link).

## File Structure

The data is organized hierarchically as follows:

```
sequence_dir/
├── prompt_dir/
│   ├── cam0.pt
│   ├── cam1.pt
│   └── ...
└── ...
```

For example:
```
tennis/
├── tennis_shot/
│   ├── cam0.pt
│   ├── cam1.pt
│   └── ...
└── ...
```

- The sequence directory name (e.g., "tennis") is the data sequence
- The prompt directory name (e.g., "tennis_shot") is the action prompt

## Data Format

Each `.pt` file is a PyTorch serialized dictionary containing segmentation masks and bounding boxes for a specific camera view, organized by the frame numbers in which the prompt occurs.</br>

Frames without detections are not included in the dictionaries

### Dictionary Structure

```
{
    "masks": {
        frame_number: segmentation_mask,
        ...
    },
    "boxes": {
        frame_number: bounding_box,
        ...
    }
}
```

- `"masks"`: A dictionary where keys are frame numbers and values are segmentation masks for the prompted action
- `"boxes"`: A dictionary where keys are frame numbers and values are bounding boxes for the prompted action

### Example

For a file `tennis/tennis_shot/cam0.pt`, the structure might be:
```
{
    "masks": {
        10: [mask_data],
        11: [mask_data],
        ...
    },
    "boxes": {
        10: [x1, y1, x2, y2],
        11: [x1, y1, x2, y2],
        ...
    }
}
```
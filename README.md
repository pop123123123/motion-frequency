# motion-frequency

Plot frequency of a video's motion over time.
Tries to find repeatitive motion and its associated frequency.

## Usage

```sh
pip3 install -r requirements.txt
./main.py VIDEO_PATH [END_FRAME [NO_SCENE_DETECTION]]
```

## Testing material
First, videos of geometric shapes moving at constant or variable frequencies were created.
Then, videos or extracts (manually chosen) from the internet (YouTube...) were used.

## Testing results
This program has been found to work well on straight movements. Rotative motions can work, but sometimes lead to an harmonic of the real ferquency.
Due to the fact that it uses the whole image to find an "average" movement, and not just some features tracked over time, this program can not find the frequencies of multiple simultaneous motions.

## Notice

This project is in a very early state. It works well as a POC, but needs great improvement.


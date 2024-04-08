# Facial Expressions Recognition
The Project is the combination of two models of Object recognition on a model found somewhere on the Internet and Emotion recognition, using YOLOv8 and [AffectNet](http://mohammadmahoor.com/affectnet/), by Mollahosseini. With the last I needed some time and patience to train the model, however, the dataset was good enough and fit the purpose.

### Run for webcam video streaming
```sh
python main.py
```

### Run for image processing in

```sh
python3 image.py 
```

It will read images from `images/input` write into `images/output`

### If you want to process a an individual

`python3 image.py /path/to/image/.jpg `

### Supported Emotions

- [0] Anger
- [1] Contempt
- [2] Disgust
- [3] Fear
- [4] Happy
- [5] Neutral
- [6] Sad
- [7] Surprise

## Demo

[demo.mkv](./docs/demo.mkv)

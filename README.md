# private_detector_model

A tensorflow implementation of <a href="https://www.theverge.com/2019/4/24/18514247/bumble-private-detector-ai-filter-lewd-images">"lewd" image detector</a>.
The presented model can classify images into two distinct categories:
 - potentially inappropriate content such as nudity and pornography
 - "normal" images

## To run:
  `python3 inference.py ./test.jpg`
## Using docker:
  `docker run -v $(pwd):/samples magiclabco/private_detector_model /samples/test.jpg`

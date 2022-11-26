# Code

Our code for large scale experiments can be found in `.py` files.
Smaller scale experiments are in notebooks `.ipynb`: those can be uploaded on Google Colab and have been thought to work out of the box.  

## Dependencies

The Deel-Lip library is among the dependencies of the the library -- we embed its wheel `wheels/deel_lip-1.2.0-py2.py3-none-any.whl`. However notice that this library can be found online [here](https://github.com/deel-ai/deel-lip).  
  
The code uses custom data loaders and data augmentation pipelines in Deep Learning Toolbox (DLT) -- we embed its wheel `wheels/DLT-0.1.0-py2.py3-none-any.whl`.   
  
Take note of **PEP427**: "A wheel is a ZIP-format archive with a specially formatted file name and the .whl extension.". 

## Experiments in Appendix

Other Pareto front in Appendix can be generated using the `pareto-front-experiment.py` with appropriate arguments.  
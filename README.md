# DTLN in C
## Project description
* This is a project aiming to create my own neural network inference library.
* This project has two parts, one is the parser part which has not yet been made public and this is the inference part which has been made public but I considered not done, I will try to spend time to make it much better.
* This program here utilized the Dual-signal Transformation LSTM Network(DTLN) to denoise audio comming in from a microphone(with miniaudio full duplex)
* This part is made public first because it can produce its own goal, and I must push my Medium story immediately so I rushed this out.
## Project structure
### Parser
* The parser is not yet made public, it's purpose is to parse an ONNX model file and create a header file similar to "model1.h", "model1.c", "model2.h", "model2.c".
* The generated header and source file contain the node description and weight of the file, it is then added into the inference part.  
* It's dependency is protobuf.
### Inference
* This project.
* It runs the DTLN model by using the pair of files created by the parser.
* There is still a lot of optimization and making it conform to coding guidelines should be done.
# Reference
1. https://github.com/breizhn/DTLN
2. https://miniaud.io/
3. https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html

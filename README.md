# DTLN in C
## Project description
* This is a project trying to create my own neural network inference library.
* This project has two parts, one is the parser part which has yet been made public and this is the inference part which is made public but I considered not done, and I will try to spend time to make it much better.
* This program here utilized the Dual-signal Transformation LSTM Network(DTLN) to denoise audio comming in from a microphone(with miniaudio full duplex)
* This part is made public first because it can produce it's own goal, and I must push my Medium story immediatedly so I rushed this out.
## Project stucture
### Parser
* The parser is not yet made public, it's purpose is to parse an ONNX model file and create a header file similar to "model1.h", "model1.c", "model2.h", "model2.c".
* The generated header and source file containes the node description and weight of the file, it is then added into the inference part.  
* It's dependecy is protobuf.
### Inference
* This project.
* It runs the DTLN model by using the pair of file created by the parser.
* There are still a lot of optimization and making it conform to codeing guideline should be done.

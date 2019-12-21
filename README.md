# GZZchat
*by Limian Guo, Xiangran Zhao, Fengzhencheng Zeng*

## Introduction
A web faced chatbot based on transformer model built with tensorflow 2.0 and flask.

The tutorial that we referred to build the model is 
[*A Transformer Chatbot Tutorial With Tensorflow 2.0*](https://medium.com/tensorflow/a-transformer-chatbot-tutorial-with-tensorflow-2-0-88bf59e66fe2).

## Instructions
### Install
You'll need python version >= 3.5 to run the program. Use `pip install -r requirements.txt` to install dependencies.

### Model
**Warning!** Train the model will overwrite the previously trained weights!

To train the model, you'll first need to build an appropriate config. An example is given as configs/default.py.
Then in run_train.py, replace `'configs.default'` with the name of your config as a python module.
Then you can run `python run_train.py` to train the model.
The trained weights will be stored in weights folder with the name given in the config.

To test the model only, replace the `'configs.default'` in run_eval.py with your config 
and replace `'Hello world!'` with any sentence you'd like to talk to the chatbot, then run `python run_eval.py`.

### Server
This chatbot is implemented in WEB stacks. The back-end is using a light weight
server Flask for easy integration with Tensorflow trained model.

You need to run `python server.py` to get the server start.
Then go to `localhost:5000` to use the chatbot.

One advantage of using a web structure is that training thread will never block user
thread which will have better user experience.
### UI
UI was built using React and integrated well with Flask so all you need to see the UI is to 
either just use what is already offered or build your own by following the instructions.

For building the UI. You need to have Node.js installed on your computer. After installation
you need to run `cd gzzchat`. Then call `npm install` to install all dependencies.

Using `npm run build` to build the UI files.


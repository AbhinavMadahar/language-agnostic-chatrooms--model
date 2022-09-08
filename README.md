# Language Agnostic Chatrooms

_The CadiLAC of Chatrooms_

This is a website which has chatrooms.
Each chatroom has a topic, e.g. spaghetti.
Anyone can join any chatroom; private chatrooms might come later.
The chatrooms are not dependant on language; there might be one user speaking Azeri and another speaking Zulu.
When users join the website, they are asked their preferred language.
The website then translates all messages written in that room to their language.
This way, we partially remove language as a barrier of communicate.

The idea for this website was inspired by a discussion I had with an Irish, German, and English traveller.
We talked about how the Internet, a mechanism by which people can easily share information, was expected to give better outcomes than what it actually gave us.
This website is a vision for a better use of the internet.

This directory implements the machine translation engine.

The machine translation engine has not yet been implemented.
However, we have already established an API for it which will not change.
To use the engine, you must start a process by running the `main.py` script.
You may then request translations by sending inputs in the format:
```
in->out: text
```
where `in` is the input language (e.g. `es`), `out` is the output language (e.g. `fr`), and `text` is the input text to be translated (e.g. `tengo un gato`).
The engine will then print to standard out the translated text; in the above example, it would be `I have a cat`.

As a note on vocabulary, a machine translation _engine_ and a machine translation _model_ are basically the same.
The term _engine_ makes more sense from the perspective of a software engineer using the system while the term _model_ makes more sense to an ML researcher designing the system.
With that in mind, this directory externally describes it as a machine translation _engine_ for the software engineer(s) who use it in the app.
Within this directory, however, it is refered to as a machine translation model, following the ML research standard.


## Code structure

The `machine-translation-engine-config.json` lets us define the machine translation engine.
It does not consider the internals of the engine, like how many layers to use in the encoder, etc.
Instead, it defines things like which languages are supported.
We keep this file in the general repository directory so that the app and the engine both use the same config file.
That way, for example, the backend knows which languages the engine supports.


## TODO

- [x] Design architecture
    - [x] Read literature
    - [x] Decide how to formulate this problem
        - How many languages should we support?
          It's easy to translate ~100 most common languages because we have a lot of data for them (e.g. Arabic, Bengali, Chinese).
          Translating less common languages is harder because there aren't as many data sets available for them.
- [x] Figure out how much storage space will be taken up by the dataset
- [ ] Select languages to target
- [ ] Download dataset
    - [ ] Write download script
        - [ ] Make download script stop using so much RAM
    - [ ] Run download script on training machine
- [ ] Implement model
    - [ ] Write code to implement encoder, decoder, and training
    - [ ] Write code to evaluate a model
    - [ ] Implement beam search
    - [ ] Run beam search to find best hyperparameter configuration
    - [ ] Train the best hyperparameter configuration fully
- [ ] Connect `main.py` to the model

## Model design

This problem can be characterized as a _many-to-many machine translation problem_.
The _many-to-many_ means that the model should be able to take one of many languages and translate to one of many languages.
For example, we might ask it to translate from Korean to Russian, or from Japanese to Portugese.

We use the architecture introduced in arXiv:2206.14982.

For simplicity, we will only support the top 50 most common languages.


## Training environment

The dataset will be on the order of around 10 gigabytes.
It might seem quite small, but it's reasonable considering that it's just a bunch of text.

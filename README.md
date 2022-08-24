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

The machine translation engine is implemented in the `machine-translation-engine` directory.
To use the engine, read the instructions in that directory's README.


## Code structure

The `machine-translation-engine-config.json` lets us define the machine translation engine.
It does not consider the internals of the engine, like how many layers to use in the encoder, etc.
Instead, it defines things like which languages are supported.
We keep this file in the general repository directory so that the app and the engine both use the same config file.
That way, for example, the backend knows which languages the engine supports.
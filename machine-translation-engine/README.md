# Machine Translation Engine

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


## TODO

- [ ] Design architecture
    - [ ] Read literature
    - [ ] Decide how to formulate this problem
        - How many languages should we support?
          It's easy to translate ~100 most common languages because we have a lot of data for them (e.g. Arabic, Bengali, Chinese).
          Translating less common languages is harder because there aren't as many data sets available for them.
- [ ] Implement model
    - [ ] ...
    - [ ] Train on more powerful hardware
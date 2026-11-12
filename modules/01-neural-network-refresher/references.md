# Module 01 — Further study

Ordered by usefulness for this module. "Level" is relative to this course's learner.

| Resource | Type | Level | What it contributes |
|---|---|---|---|
| Andrej Karpathy, [*The spelled-out intro to neural networks and backpropagation: building micrograd*](https://www.youtube.com/watch?v=VMj-3S1tku0) (2022) | video, 2.5 h | introductory | Builds a scalar autograd engine from nothing. The best way to make "autograd is just the chain rule, recorded" concrete. |
| Andrej Karpathy, [*Building makemore Part 2: MLP*](https://www.youtube.com/watch?v=TCH_1BHY58I) and [*Part 4: Becoming a Backprop Ninja*](https://www.youtube.com/watch?v=q8SA3rM6ckI) (2022) | video | introductory–intermediate | Part 2 is the window-MLP language model of section D.7 in PyTorch; Part 4 hand-derives the backward pass of a deeper network, including cross-entropy. Do Part 4 after the Challenge exercise. |
| Bengio, Ducharme, Vincent, Jauvin, [*A Neural Probabilistic Language Model*](https://www.jmlr.org/papers/v3/bengio03a.html), JMLR 2003 | paper | intermediate | The origin of section D.7: learned word vectors + an MLP over a fixed window. Read sections 1–2. Historically important; the architecture itself is superseded. |
| Rumelhart, Hinton, Williams, [*Learning representations by back-propagating errors*](https://www.nature.com/articles/323533a0), Nature 1986 | paper | historical | The paper behind the networks you trained in the 1990s. Four pages; worth rereading with today's eyes. |
| Goodfellow, Bengio, Courville, [*Deep Learning*](https://www.deeplearningbook.org/), ch. 6 "Deep Feedforward Networks" (2016) | book (free online) | intermediate | Rigorous treatment of MLPs, output units, why cross-entropy pairs with softmax, and backprop as an algorithm on computation graphs. |
| Michael Nielsen, [*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/), ch. 1–3 (2015) | book (free online) | introductory | Digit recognition with an MLP, backprop derivation, and a clear explanation of why cross-entropy learns faster than squared error with sigmoid outputs. Closest in spirit to your original experience. |
| PyTorch documentation, [*A Gentle Introduction to torch.autograd*](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) | documentation | introductory | The mechanics behind `loss.backward()` and `requires_grad`. |

Historical versus current: the 1986 and 2003 papers are foundations, not descriptions of
current practice. Current practice for the *same ideas* is what you see in nanochat:
autograd, no biases, wider hidden layers, different non-linearities, and adaptive
optimizers.

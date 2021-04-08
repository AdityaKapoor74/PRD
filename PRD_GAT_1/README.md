<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#arch1">Architecture 1</a>
    </li>
    <li>
      <a href="#arch2">Architecture 2</a>
    </li>
    <li>
      <a href="#arch3">Architecture 3</a>
    </li>
  </ol>
</details>



<!-- ARCHITECTURE 1 -->
## Architecture 1

In this methodology we have 3 networks for the Critic Architecture stacked together:-
1) Scalar Dot Product Attention to do the preprocessing of the observations of every agent with a message passing round (we use raw observations to calculate the weights) (SOFTMAX to calculate the weights)
2) Scalar Dot Product Attention to calculate the z values (We use raw observations to calculate the weights; sometimes a small subset of the raw observations is also used) (SIGMOID to calculate the weights so that there is a pair-wise dependency)
3) Fully Connected Network to calculate the Value estimates for an agent i not conditioned on agent j's actions; output is a NxN matrix where N is the number of agents


## Architecture 2

In this methodology we have 3 networks for the Critic Architecture stacked together:-
1) Scalar Dot Product Attention to calculate the z values (We use raw observations to calculate the weights) (SIGMOID to calculate the weights so that there is a pair-wise dependency)
2) Scalar Dot Product Attention to do the message passing of the observations of every agent to every other agent; sharing of information (we use raw observations concatenated with z values to calculate the weights) (SOFTMAX to calculate the weights)
3) Fully Connected Network to calculate the Value estimates for an agent i not conditioned on agent j's actions; output is a NxN matrix where N is the number of agents



## Architecture 3

In this methodology we have 2 networks for the Critic Architecture stacked together:-
1) Scalar Dot Product Attention to do the message passing of the observations of every agent to every other agent; sharing of information (we use raw observations concatenated with action values to calculate the weights) (SIGMOID/SOFTMAX to calculate the weights)
2) Fully Connected Network to calculate the Value estimates for an agent i not conditioned on agent j's actions; output is a NxN matrix where N is the number of agents
# Commit by commit recall

* ### Pre git init
**Date:** December 2024

#### The Goal: What was I trying to achieve with this change?
Implemented a slow game environment for 2048, it used many loops and was extremely inefficient.


#### The Challenge: Was this harder or easier than I expected? Why?
Harder: The logic for moving and merging tiles in 2048 was more complex than I initially anticipated, especially when handling edge cases like multiple merges in a single move.

#### The "Aha!" Moment: Did I learn or realize anything important?
Aha! I realized that I could simplify the movement logic by breaking it down into smaller, reusable functions for each direction, which made the code cleaner and easier to debug.

#### Story Relevance: Is this a key "plot point" in the project's story?
Supporting Detail: This initial implementation laid the groundwork for the project, but it was not the most efficient or elegant solution.

* ### Commit #1
**Date:** 18/12/2024

**Hash:** 5b9b2d882ae082833e216d0796fe2dc51acaf7ff

#### The Goal: What was I trying to achieve with this change?
1. Implement a fast game environment for 2048 using NumPy and Look Up Tables (LUTs) to optimize the move operations. This LUT were just made for one move to the left, and the other moves were implemented by rotating and flipping this row. Each row of the board was represented as a 16-bit integer, allowing for efficient indexing into the LUTs.
2. Used this environment to create a OpenAI Gym environment for 2048, enabling the use of reinforcement learning algorithms to train agents to play the game.
3. Developed the first Deep Q-Network (DQN) agent using PyTorch to play 2048. The neural network consisted of three fully connected hidden layers with ReLU activation functions, mapping the board state to Q-values for each possible action. The agent was trained using experience replay.The optimizer used was AdamW. The training loop followed the epsilon-greedy policy for exploration, and the loss function was the Huber (Smooth L1) loss.
4. The inputs where  the board flattened, the max tile reached, the empty cells, and the sum of the tiles. The reward function was the move score and the empty cells weighted, the model was penalized for doing nothing.
5. Trained 3 simple models, best one could barely get 256 tiles.

#### The Challenge: Was this harder or easier than I expected? Why?
1. Harder: Understanding how to effectively use LUTs for the game logic was challenging. It required a deep understanding of bit manipulation and how to represent the game state efficiently.
2. Harder: Implementing the OpenAI Gym environment required a good grasp of the Gym API and how to structure the environment to be compatible with reinforcement learning algorithms.
3. Harder: Designing and training the DQN agent was more complex than I initially thought. It involved tuning hyperparameters, managing the replay buffer, and ensuring the stability of the learning process with a target network, I had no idea how anything worked, so I just tinkered with it.
4. Harder: I had no idea how to create a good reward structure, so I just made something up and hoped for the best.
5. Harder: Training the models to achieve higher tile values was more difficult than expected. The agent struggled to learn effective strategies, often getting stuck at lower tile values like 256.



#### The "Aha!" Moment: Did I learn or realize anything important?
1. Aha! I realized that by precomputing the results of all possible row configurations and storing them in LUTs, I could drastically reduce the computational overhead during gameplay. This approach allowed for O(1) time complexity for move operations, making the game environment significantly faster.
2. Aha! I learned that structuring the environment to follow the Gym API conventions made it easier to integrate with existing reinforcement learning libraries and tools.
3. Aha! I discovered that using experience replay and a target network were crucial for stabilizing the training of the DQN agent. 
4. Aha! I realized that the choice of input features had a significant impact on the agent's ability to learn. Including features like the maximum tile, number of empty cells, and sum of tiles provided the agent with more context about the game state.
5. Aha! I realized that the choice of reward structure and exploration strategy had a significant impact on the agent's ability to learn effective strategies for achieving higher tile values.

#### Story Relevance: Is this a key "plot point" in the project's story?
1. Key Moment: This commit was a major turning point in the project, as it transformed the game environment from a slow, loop-based implementation to a highly efficient one using advanced techniques like LUTs and bit manipulation. This optimization was crucial for enabling more complex AI algorithms to interact with the game in real-time.
2. Key Moment: Creating the OpenAI Gym environment was essential for leveraging reinforcement learning techniques, which are central to the project's goal of developing an AI agent to play 2048.
3. Key Moment: The implementation of the DQN agent marked the beginning of the AI component of the project, setting the stage for further experimentation with different architectures and training strategies.
4. Supporting Detail: The choice of input features and reward structure were important considerations that influenced the agent's learning process, but they were not the primary focus of this commit.
5. Supporting Detail: While the initial models struggled to achieve high tile values, this phase of the project was important for understanding the challenges of training AI agents in a complex environment like 2048.

* ### Commit #2
**Date:** 30/08/2025

**Hash:** e8947e5e705259455de99167be1dbcd8035a5218

#### The Goal: What was I trying to achieve with this change?
Getting back in the project as it was left for 9 months, refactored the Fast2048 code, putting the LUTs inside the class adn fixing some minor bugs. Changed the NN architecture to three 1024-neuron hidden layers.

#### The Challenge: Was this harder or easier than I expected? Why?
Harder: Refactoring the code after a long break was challenging, as I had to reacquaint myself with the existing implementation and understand the nuances of the LUT-based approach.

#### The "Aha!" Moment: Did I learn or realize anything important?
Aha! I realized that encapsulating the LUTs within the class improved code organization and made it easier to manage the game logic. This change also facilitated future modifications and enhancements to the environment.

#### Story Relevance: Is this a key "plot point" in the project's story?
Supporting Detail: This commit was more about maintenance and code quality rather than introducing new features or significant changes to the project's direction.

* ### Commit #3
**Date:** 30/08/2025

**Hash:** 0a0a0fe86f248081bee61de1c4b51ea80101fefd

#### The Goal: What was I trying to achieve with this change?
Implemented Wandb logging to track training progress and performance metrics. This included logging rewards, and other relevant statistics to visualize the agent's learning over time. Also implemented prefilling the replay buffer with random moves to provide the agent with a diverse set of experiences from the start of training.

#### The Challenge: Was this harder or easier than I expected? Why?
Harder: Integrating Wandb into the existing training loop required careful consideration of what metrics to log and how to structure the logging calls to avoid performance bottlenecks during training.

#### The "Aha!" Moment: Did I learn or realize anything important?
Aha! I realized that pre-filling the replay buffer with random moves helped to stabilize the initial training phase by providing the agent with a broader range of experiences, which improved its ability to learn effective strategies from the outset.

#### Story Relevance: Is this a key "plot point" in the project's story?
Supporting Detail: While this commit improved the training process and monitoring capabilities, it was more of an enhancement rather than a fundamental change to the project's core objectives.

* ### Commit #4
**Date:** 31/08/2025

**Hash:** cc830f17e2eea47b92f0ca30595ee46b2aac5447

#### The Goal: What was I trying to achieve with this change?
Changed to a ConvDQN architecture with 16-channel board input (one-hot encoding of tile exponents), input normalized to \[0,1\] with shape (16, 4, 4), and updated hyperparameters: BATCH\_SIZE=1024, GAMMA=0.99, EPS\_START=0.9, EPS\_END=0.05, EPS\_DECAY=1,000,000, TAU=0.005, LR=1e-4, UPDATE\_FREQUENCY=4, MEMORY\_SIZE=100,000, DEVICE="cuda".

#### The Challenge: Was this harder or easier than I expected? Why?
Harder: Transitioning to a convolutional architecture required a solid understanding of how to effectively process spatial data, as well as tuning the hyperparameters to ensure stable and efficient learning.

#### The "Aha!" Moment: Did I learn or realize anything important?
Aha! I realized that using a one-hot encoding for the board state allowed the convolutional layers to better capture spatial relationships between tiles, which improved the agent's ability to learn effective strategies for playing 2048, however it still barely got to 256.

#### Story Relevance: Is this a key "plot point" in the project's story?
Supporting Detail: This commit represented a significant shift in the model architecture, which was an important step in exploring different approaches to training the agent, but it did not yet lead to substantial improvements in performance.

* ### Commit #5
**Date:** 31/08/2025

**Hash:** f03e5a7e7aea932fd12966391b089b608a2f3b40

#### The Goal: What was I trying to achieve with this change?
Minor refactor Fast2048 

#### The Challenge: Was this harder or easier than I expected? Why?
Easier: The refactor was straightforward, as it mainly involved cleaning up the code and improving readability without altering the core functionality.

#### Story Relevance: Is this a key "plot point" in the project's story?
Supporting Detail: This commit was focused on code quality and maintainability rather than introducing new features or significant changes to the project's direction.

* ### Commit #6
**Date:** 31/08/2025

**Hash:** 80740fa196ff81153ddaad159d7ca27b2d15fe81

#### The Goal: What was I trying to achieve with this change?
Refactored the codebase for modularity and implemented a PPO agent using Stable Baselines3. Introduced a custom CNN feature extractor for the agent, leveraging convolutional layers to process the board state. Training was managed with vectorized environments, extensive Wandb logging, and periodic checkpointing. Additionally, a Pygame-based visualizer was added to render the agent's gameplay, allowing real-time observation of the PPO agent's decisions and performance.

#### The Challenge: Was this harder or easier than I expected? Why?
Harder: Refactoring the codebase for modularity required careful planning to ensure that components were well-defined and could be easily reused. Implementing the PPO agent with Stable Baselines3 involved understanding the library's API and how to customize the feature extractor to suit the 2048 game environment. Integrating vectorized environments and setting up comprehensive logging and checkpointing added additional layers of complexity to the training process. Finally, developing a Pygame-based visualizer required knowledge of graphical rendering and event handling to create an interactive experience.

#### The "Aha!" Moment: Did I learn or realize anything important?
Aha! I realized that using Stable Baselines3 significantly streamlined the implementation of the PPO agent, allowing me to focus more on customizing the feature extractor and training process rather than building the algorithm from scratch. The modular codebase made it easier to manage different components of the project, and the visualizer provided valuable insights into the agent's behavior, which helped in debugging and refining the training strategy.

#### Story Relevance: Is this a key "plot point" in the project's story?
Key Moment: This commit marked a major advancement in the project, transitioning from a DQN-based approach to a more sophisticated PPO agent. The introduction of modularity, comprehensive logging, and a visualizer significantly enhanced the project's structure and usability, setting the stage for more effective training and evaluation of the AI agent.

* ### Fix #1 (Infinite Search Loop)
**Date:** 05/05/2026

#### The Goal: What was I trying to achieve with this change?
Debug and fix a hard hang in the C++ Expectimax searcher where it would enter an infinite loop on specific board states.

#### The Challenge: Was this harder or easier than I expected? Why?
Extremely Hard: The bug was caused by a "perfect storm" of pure hash collisions in the Transposition Table and I/O blocking from debug logs. It required deep tracing of the search passes to realize that the searcher was losing its progress because two "hot" keys were fighting for the same bucket.

#### The "Aha!" Moment: Did I learn or realize anything important?
Aha! I realized that in a multi-pass deferred-batching searcher, the Transposition Table is not just an optimization—it's the primary mechanism for termination. If progress is overwritten, the searcher can loop forever. I also learned that 4-way associativity is a "magic number" for C++ performance, as it fits perfectly into a 64-byte cache line.

#### Story Relevance: Is this a key "plot point" in the project's story?
Key Breakthrough: This fix stabilized the C++/Python hybrid searcher, allowing it to scale to arbitrary depths without hanging. It also established the "C++ Searcher Development Guidelines" for future improvements.

---
# Sources
https://github.com/tsangwpx/ml2048/tree/main
https://arxiv.org/abs/2212.11087
https://www.youtube.com/watch?v=9gQQAO4I1Ck
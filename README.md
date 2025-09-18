Machine Learning Overview

Machine Learning (ML) is a subset of Artificial Intelligence (AI) that enables systems to learn patterns from data and improve performance over time without being explicitly programmed. It draws inspiration from biological brains, statistics, and optimization techniques, powering many of today’s applications in technology, science, and industry.

🧠 Artificial Neural Networks (ANNs)

ANNs are inspired by the structure of the human brain.

They consist of artificial neurons connected by edges (synapses).

Each connection carries a signal (a real number), and neurons compute outputs using non-linear functions of the sum of inputs.

Connections have weights, which adjust during learning to increase or decrease signal strength.

Neurons may also have thresholds, activating only when inputs cross a limit.

Organized into layers:

Input layer – receives raw data.

Hidden layers – perform transformations.

Output layer – produces predictions.

Deep Learning: uses multiple hidden layers to mimic human processing of vision, sound, and language.

Applications: computer vision, speech recognition, medical diagnosis, machine translation, and more.

🌳 Decision Trees

Decision tree learning uses a tree-like model for predictions.

Classification Trees: output discrete values (class labels).

Regression Trees: output continuous values.

Leaves: represent outcomes (labels).

Branches: represent feature-based decisions.

Widely used in statistics, data mining, and decision analysis.

📊 Support Vector Machines (SVMs)

Supervised learning models used for classification and regression.

Build a linear decision boundary (hyperplane) between two classes.

Can perform non-linear classification using the kernel trick, which maps inputs to high-dimensional spaces.

Probabilistic classification possible with methods like Platt scaling.

📈 Regression Analysis

Explores the relationship between input variables and outcomes.

Linear Regression: fits a straight line using methods like Ordinary Least Squares (OLS).

Regularized Regression: ridge, lasso – reduce overfitting.

Non-linear models:

Polynomial Regression

Logistic Regression (for classification)

Kernel Regression

🔗 Bayesian Networks

Probabilistic graphical models represented by Directed Acyclic Graphs (DAGs).

Nodes = random variables, Edges = conditional dependencies.

Example: Rain → Sprinkler → Wet Grass.

Applications: medical diagnosis, speech processing, protein sequence modeling.

Variants:

Dynamic Bayesian Networks (sequences over time).

Influence Diagrams (decision problems under uncertainty).

🧬 Genetic Algorithms (GAs)

Inspired by natural selection.

Uses mutation, crossover, and selection to evolve better solutions.

Historically popular in the 1980s–90s.

ML techniques now help optimize genetic algorithms themselves.

🎓 Training Models

Requires large and representative datasets.

Common sources: text corpora, image collections, user behavior data.

Risk: overfitting (memorizing data instead of generalizing).

🌍 Federated Learning

Distributed form of training where data stays on local devices.

Improves privacy (no central data storage).

Increases efficiency by using edge devices.

Example: Google Gboard predicts queries using federated ML.

🚀 Applications of Machine Learning

Agriculture

Banking & Insurance

Bioinformatics & Healthcare

Computer Vision & Speech Recognition

Natural Language Processing & Translation

Recommendation Systems (e.g., Netflix, Amazon)

Fraud Detection (credit card, online transactions)

Finance & Market Prediction

Robotics & Autonomous Systems

User Behavior Analytics

⚠️ Limitations

Failures due to:

Insufficient or biased data

Privacy concerns

Wrong choice of algorithms/tools

Lack of resources/expertise

Examples:

Uber’s self-driving car accident (2018).

IBM Watson’s failures in healthcare despite billions invested.

🎭 Bias in Machine Learning

ML models often inherit human biases from training data.

Risks:

Discrimination in hiring, policing, healthcare.

Misclassification of minorities (e.g., Google Photos incident).

Ethical concerns:

Need for fairness-aware algorithms.

Responsible data collection.

Transparency in decision-making.

📏 Model Assessment

Validation techniques:

Holdout method (train/test split).

K-Fold Cross-Validation.

Bootstrap sampling.

Metrics:

Accuracy, Precision, Recall, F1-score.

ROC-AUC, TOC (Total Operating Characteristic).

⚖️ Ethics in Machine Learning

Key concerns:

Bias (racial, gender, cultural).

Profit-driven models in healthcare.

Misuse in surveillance & privacy violations.

Reminder: AI impacts people directly, so ethical responsibility is essential.

🖥️ Hardware & Software

Hardware:

Shift from CPUs to GPUs/TPUs for deep learning.

Compute requirements growing exponentially (e.g., AlphaZero vs. AlexNet).

Software:

Popular ML frameworks: TensorFlow, PyTorch, Scikit-learn, Keras.

📚 Conclusion

Machine Learning is a powerful paradigm transforming industries from healthcare to finance. While it offers massive opportunities, it also comes with challenges like bias, ethical risks, and high computational demands. With responsible use and continual improvement, ML has the potential to shape a more intelligent and fair future.

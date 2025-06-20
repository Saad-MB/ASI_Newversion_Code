import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------------
# 1. Enhanced Data Preparation
# ----------------------------
def prepare_data():
    data = load_breast_cancer()
    X, y = data.data, data.target
    # Convert to binary classification: malignant=1, benign=0
    y = np.where(y == 0, 1, 0)
    X = StandardScaler().fit_transform(X)
    X = np.hstack([X, np.ones((X.shape[0], 1))])  # Intercept
    return X.astype(np.float32), y.astype(np.float32)

X, y = prepare_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# 2. Numerically Stable Log-Probability
# ----------------------------
prior_variance = 10.0  # More reasonable prior

def log_prob_logistic(params, batch_data):
    X_batch = batch_data[:, :-1]
    y_batch = batch_data[:, -1]
    logits = tf.tensordot(X_batch, params, axes=1)
    
    # Numerically stable log-likelihood
    max_logits = tf.clip_by_value(logits, -50, 50)  # Prevent overflow
    log_likelihood = tf.reduce_sum(
        y_batch * max_logits - tf.math.softplus(max_logits)
    )
    
    # Regularization with clipping
    params_clipped = tf.clip_by_norm(params, 10.0)  # Prevent extreme values
    log_prior = -0.5 * tf.reduce_sum(params_clipped**2) / prior_variance
    
    # Scale to full dataset
    scaling = len(y_train) / len(y_batch)
    return scaling * log_likelihood + log_prior

# ----------------------------
# 3. Robust SGHMC Sampler
# ----------------------------
def stable_sghmc_sampler(log_prob_func, initial_params, num_samples, learning_rate, friction, batch_size, data):
    params = tf.Variable(initial_params, dtype=tf.float32)
    momentum = tf.Variable(tf.zeros_like(params), dtype=tf.float32)
    sampled_params = []
    num_data = data.shape[0]
    
    # Adaptive learning rate decay
    lr_decay = tf.optimizers.schedules.ExponentialDecay(
        learning_rate, num_samples//10, 0.95
    )
    
    for step in range(num_samples):
        idx = np.random.choice(num_data, batch_size, replace=False)
        batch = data[idx]
        
        with tf.GradientTape() as tape:
            neg_log_prob = -log_prob_func(params, batch)
        
        grad = tape.gradient(neg_log_prob, params)
        
        # Gradient clipping
        clipped_grad = tf.clip_by_norm(grad, 5.0)
        
        # Adaptive learning rate
        current_lr = lr_decay(step)
        
        # Correct physics-inspired update
        noise = tf.random.normal(shape=params.shape, stddev=tf.sqrt(2 * current_lr * friction))
        momentum.assign(
            (1 - friction) * momentum - 
            current_lr * clipped_grad +
            noise
        )
        params.assign_add(momentum)
        
        # Record only finite values
        if tf.math.reduce_all(tf.math.is_finite(params)):
            sampled_params.append(params.numpy())
        
        if step % 1000 == 0:
            param_norm = tf.norm(params).numpy()
            grad_norm = tf.norm(grad).numpy() if grad is not None else 0
            print(f"Step {step}/{num_samples}, LR: {current_lr:.2e}, "
                  f"Param norm: {param_norm:.2f}, Grad norm: {grad_norm:.2f}")
    
    return np.array(sampled_params)

# ----------------------------
# 4. Run with Stabilized Settings
# ----------------------------
# Conservative hyperparameters
initial_params = np.zeros(X_train.shape[1])
num_samples = 10000
learning_rate = 5e-5  # Reduced from 1e-4
friction = 0.05       # Reduced friction
batch_size = 100

print("Starting STABLE SGHMC sampling...")
samples = stable_sghmc_sampler(
    log_prob_logistic,
    initial_params,
    num_samples,
    learning_rate,
    friction,
    batch_size,
    np.hstack([X_train, y_train.reshape(-1, 1)]))
print(f"Obtained {len(samples)} stable samples")

# Burn-in removal (first 20%)
posterior_samples = samples[len(samples)//5:]

# ----------------------------
# 5. Analysis and Diagnostics
# ----------------------------
# 1. Trace plots
plt.figure(figsize=(12, 6))
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(posterior_samples[:, i])
    plt.title(f'Trace Plot - Weight {i+1}')
    plt.xlabel('Iteration')
plt.tight_layout()
plt.savefig('stable_traces.png', dpi=300)
plt.show()

# 2. Posterior distributions
plt.figure(figsize=(10, 6))
plt.hist(posterior_samples[:, 0], bins=30, density=True, alpha=0.7)
plt.title('Posterior Distribution - Weight 1')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True)
plt.savefig('stable_posterior.png', dpi=300)
plt.show()

# 3. Predictive performance
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

posterior_mean = np.mean(posterior_samples, axis=0)
y_prob = sigmoid(X_test @ posterior_mean)
y_pred = (y_prob > 0.5).astype(int)
accuracy = np.mean(y_pred == y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")

# 4. Uncertainty analysis
predictive_samples = []
for i in range(0, len(posterior_samples), 100):
    y_prob_sample = sigmoid(X_test @ posterior_samples[i])
    predictive_samples.append(y_prob_sample)
    
predictive_mean = np.mean(predictive_samples, axis=0)
predictive_std = np.std(predictive_samples, axis=0)

# Plot reliability diagram
bins = np.linspace(0, 1, 11)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_acc = []
for i in range(len(bins)-1):
    in_bin = (predictive_mean >= bins[i]) & (predictive_mean < bins[i+1])
    if np.any(in_bin):
        bin_acc.append(np.mean(y_test[in_bin] == (predictive_mean[in_bin] > 0.5)))
    else:
        bin_acc.append(0)

plt.figure(figsize=(8, 6))
plt.plot(bin_centers, bin_acc, 'o-', label='SGHMC')
plt.plot([0,1], [0,1], 'k--', label='Perfect calibration')
plt.xlabel('Mean Predictive Probability')
plt.ylabel('Actual Accuracy')
plt.title('Calibration Curve')
plt.legend()
plt.grid(True)
plt.savefig('calibration_curve.png', dpi=300)
plt.show()

print("\nSampling Statistics:")
print(f"Stable Samples: {len(samples)}/{num_samples} ({100*len(samples)/num_samples:.1f}%)")
print(f"Parameter Norm: {np.linalg.norm(posterior_mean):.2f} ± {np.std([np.linalg.norm(s) for s in posterior_samples]):.2f}")

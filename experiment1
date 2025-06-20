import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def sghmc_sampler(log_prob_func, initial_params, num_samples, learning_rate, friction_term, momentum_decay, batch_size, data):
    params = tf.Variable(initial_params, dtype=tf.float32)
    momentum = tf.Variable(tf.zeros_like(params), dtype=tf.float32)
    sampled_params = []
    num_data_points = data.shape[0]
    
    for step in range(num_samples):
        batch_indices = np.random.choice(num_data_points, batch_size, replace=False)
        current_batch = data[batch_indices]
        
        with tf.GradientTape() as tape:
            negative_log_prob = -log_prob_func(params, current_batch)
        stochastic_grad = tape.gradient(negative_log_prob, params)
        
        noise_std = np.sqrt(2 * learning_rate * friction_term)
        noise = tf.random.normal(shape=tf.shape(params), mean=0.0, stddev=noise_std)
        
        # Corrected momentum update: (1 - friction_term * learning_rate) 
        momentum.assign(
            (1 - friction_term * learning_rate) * momentum - 
            learning_rate * stochastic_grad + 
            noise
        )
        # Parameter update (outside gradient tape)
        params.assign_add(momentum)
        
        sampled_params.append(params.numpy())
        if step % 1000 == 0:
            print(f"Step {step}/{num_samples}, Current Params: {params.numpy()}")
    
    return np.array(sampled_params)

# Fixed synthetic_log_prob
def synthetic_log_prob(params, batch_data):
    mean = tf.constant([0.0, 0.0], dtype=tf.float32)
    covariance = tf.constant([[1.0, 0.5], [0.5, 1.0]], dtype=tf.float32)
    cov_inv = tf.linalg.inv(covariance)
    det_cov = tf.linalg.det(covariance)
    diff = params - mean
    
    # Correct quadratic form using einsum
    quadratic = tf.einsum('i,ij,j', diff, cov_inv, diff)
    
    # Fixed constant term: -log(2π) for 2D Gaussian
    log_prob = -0.5 * quadratic - 0.5 * tf.math.log(det_cov) - tf.math.log(tf.constant(2 * np.pi, dtype=tf.float32))
    return log_prob

# Generate dummy data
dummy_data = np.random.randn(1000, 2).astype(np.float32)

# Parameters
initial_params = np.array([-2.0, 2.0])
num_samples = 10000
learning_rate = 0.01
friction_term = 0.1
momentum_decay = 0.9  # Unused in corrected momentum update
batch_size = 100

print("Starting SGHMC sampling...")
sghmc_samples = sghmc_sampler(
    synthetic_log_prob, 
    initial_params, 
    num_samples, 
    learning_rate, 
    friction_term, 
    momentum_decay, 
    batch_size, 
    dummy_data 
)
print("SGHMC sampling complete.")

# Plotting
plt.figure(figsize=(10, 5))
plt.scatter(sghmc_samples[:, 0], sghmc_samples[:, 1], alpha=0.3, s=1)
plt.title('SGHMC Samples from a 2D Gaussian')
plt.xlabel('Parameter 1')
plt.ylabel('Parameter 2')
plt.grid(True)
plt.axvline(0, color='red', linestyle='--', label='True Mean 0')
plt.axhline(0, color='red', linestyle='--')
plt.legend()
plt.show()

# Analysis
sampled_mean = np.mean(sghmc_samples, axis=0)
sampled_covariance = np.cov(sghmc_samples, rowvar=False)
print(f"\nTrue Mean: [0.0, 0.0]")
print(f"Sampled Mean: {sampled_mean}")
print(f"True Covariance:\n[[1.0, 0.5]\n [0.5, 1.0]]")
print(f"Sampled Covariance:\n{sampled_covariance}")

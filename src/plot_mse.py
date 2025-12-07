import sys
import re
import matplotlib.pyplot as plt
import time

# Read the epoch and MSE from stdin
data = sys.stdin.read().strip().split('\n')
epochs = []
mses = []
for line in data:
    match = re.match(r'Epoch (\d+) completed. MSE: ([\d.]+)', line)
    if match:
        epochs.append(int(match.group(1)))
        mses.append(float(match.group(2)))
    else:
        print(f"Warning: Line did not match expected format: {line}", file=sys.stderr)
        
# Plot the MSE over epochs
plt.plot(epochs, mses)
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE over Epochs')
plt.grid(True)
time.sleep(1)  # Pause for a moment to ensure the plot is rendered
plt.savefig('mse_plot.png')
plt.show()
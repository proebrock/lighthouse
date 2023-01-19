import numpy as np
import matplotlib.pyplot as plt


# Number of pixels
m = 1000
# Number of images
n = 3

# Generate stack of "images"
signals = np.zeros((n, m), dtype=np.uint8)
phis = np.linspace(0, 2 * np.pi, n + 1)[:-1]
# Avoid ambiguity of angles[0] == angles[-1] # TODO: 2*pi? 4*pi? 10*pi? Ambiguity
angles = np.linspace(0, 2*np.pi, m + 1)[1:]
for i, phi in enumerate(phis):
    signal = np.sin(angles + phi + 3*np.pi/2) # TODO: 3*np.pi/2 is magic. understand and fix it!
    signal = (255.0 * (signal + 1.0)) / 2.0
    signals[i, :] = signal.astype(np.uint8)

# Modify signal to check robustness of reconstruction: offset, factor, noise, ...
#signals = (signals - 150.0) * 2.0
#signals += np.random.randint(0, 10, signals.shape)
if False:
    # Apply gamma
    signals = signals / 255.0
    signals = np.power(signals, 1.1)
    signals = (signals * 255.0).astype(np.uint8)

# Analyze stack of "images"
spectrums = np.fft.fft(signals, axis=0)
phases = np.angle(spectrums)
indices = (m * (phases[1, :] + np.pi)) / (2*np.pi)
indices = indices.round().astype(int)
indices -= 1

# Residuals of reconstruction as a quality measure
signal_reconstructed = np.fft.ifft(spectrums, axis=0).real
residuals = signal_reconstructed - signals
residual_rms = np.sqrt(np.mean(np.square(residuals)))
print(f'Residual RMS {residual_rms}')
# TODO: This is a bad quality measure of the reconstruction;
# is there anything better?

# Visualize
fig = plt.figure()
ax = fig.add_subplot(121)
for i in range(signals.shape[0]):
    ax.plot(signals[i, :], label=f'{np.rad2deg(phis[i]):.1f}')
ax.legend()
ax.grid()
ax.set_title('Signals')
ax = fig.add_subplot(122)
for i in range(signals.shape[0]):
    ax.plot(signal_reconstructed[i, :], label=f'{np.rad2deg(phis[i]):.1f}')
ax.legend()
ax.grid()
ax.set_title('Reconstructed Signals')

fig = plt.figure()
ax = fig.add_subplot(121)
for i in range(signals.shape[0]):
    ax.plot(signals[i, :], label=f'{np.rad2deg(phis[i]):.1f}')
ax.legend()
ax.grid()
ax.set_title('Generated patterns')
ax = fig.add_subplot(122)
if False:
    # Show all phases
    for i in range(phases.shape[0]):
        ax.plot(np.rad2deg(angles), phases[i, :], label=f'{i}')
else:
    # Show phase with index 1
    ax.plot(np.rad2deg(angles), phases[1, :], '-b')
ax.set_title('Analyzed images')
ax.grid()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(indices - np.arange(m))
ax.grid()
ax.set_title('Indices errors')
plt.show()

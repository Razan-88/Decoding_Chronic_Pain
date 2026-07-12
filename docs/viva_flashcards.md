\# Viva Flashcards — Generator A (synthetic recording)



Q\&A for quick recall. Detailed versions live in notebooks/01\_generator.ipynb.



\---



\*\*Q: Why build synthetic data at all?\*\*

A: To benchmark denoisers I need the true clean signal to compare against. Real recordings

never come with a clean reference, so I build the clean signal myself, add noise, denoise,

and compare back.



\*\*Q: Why a Difference of Gaussians (DoG) template instead of Ricker/Mexican-hat?\*\*

A: Real extracellular APs are asymmetric (sharp deep trough, smaller slower rebound). Ricker

is symmetric and can't match that. DoG builds trough and rebound as two separate Gaussians I

control independently. (Bonus: scipy.signal.ricker was removed in recent SciPy; DoG is just np.exp.)



\*\*Q: What does `peak\_ratio` control?\*\*

A: How big the upward rebound is relative to the downward trough. Higher = more symmetric,

lower = almost pure dip. It sets the asymmetry.



\*\*Q: What does the channel map (`amp`) represent physically?\*\*

A: One fibre fires; electrode contacts closer to it record a stronger signal, farther ones

weaker, distant ones nothing. So `amp` is the fibre's spatial footprint across the electrode.



\*\*Q: Why does the same spike appear at different strengths on different channels?\*\*

A: The electrical signal fades with distance from the fibre, so each contact's amplitude

depends on its position relative to the firing fibre.



\*\*Q: Why is the cross-channel pattern important for spike sorting?\*\*

A: Each fibre has a unique amplitude pattern across the 32 channels — a spatial fingerprint.

Sorting tells fibres apart by comparing these patterns, not just spike shape. More channels =

more distinctive (my result: F1 \~0.05 on 1 channel → \~0.51 with 3–5 channels).



\*\*Q: What goes wrong if you denoise each channel separately? (KEY)\*\*

A: Per-channel denoising treats channels independently, so it doesn't preserve the

cross-channel amplitude ratios — and those ratios are exactly what sorting relies on. That's

why I argue for joint (multi-channel) denoising, which cleans all channels together and can

preserve the pattern.



\*\*Q: What is the `latency` in `spike\_times = ttl\_times + latency`?\*\*

A: The travel delay from stimulus (TTL) to the recording electrode. It's fixed because it's

the same fibre responding to the same stimulus each time. (Also relates to fibre conduction

speed — slower fibres like some nociceptors have longer latency.)



\*\*Q: Why do evoked spikes give clean ground truth?\*\*

A: I define the spike times (TTL time + latency), so I know exactly when each spike should

appear — I'm not detecting them, I'm setting them.



\*\*Q: The double-stamp bug — what happened?\*\*

A: The stamping line uses `+=` (accumulate). Re-running the loop without re-zeroing `rec`

stacked spikes on themselves, doubling amplitudes (ch19 trough −2.0 instead of −1.0). Fix:

zero the canvas inside the same cell. Rule: any `+=` cell is unsafe to re-run alone; exact

2×/3× values are the tell.



\*\*Q: What does `noise\_std` relate to, and why sweep it?\*\*

A: It sets noise size vs spike depth (SNR). Same noise makes strong channels easy and weak

channels hard. Sweeping it lets me plot denoiser performance vs SNR instead of one number.


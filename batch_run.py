import subprocess

datasets = [
    'landcover', 'optdigits', 'page-blocks', 'ring', 'segment', 'texture',
    'twonorm', 'vehicle', 'vowel', 'winequality-red', 'winequality-white',
    'satimage', 'steel', 'spambase', 'letter', 'avila', 'magic', 'penbased',
    'biodeg', 'waveform'
]

# method, lambda
methods = [
    ('al',     0.0),   # conventional AL
    ('al_ei',  0.0),   # our method with lambda=0
]

# for repetition (just set to 0 here)
rep = 0

for dataname in datasets:
    for method, lambda_val in methods:
        cmd = [
            'python', 'run_experiment.py',
            '--dataname',  dataname,
            '--method',    method,
            '--lambda_val', str(lambda_val),
            '--rep',        str(rep)
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

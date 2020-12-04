class Params:
    # the most important parameter
    random_seed = 501105

    # system params
    verbose = True
    num_workers = 8
    device = None  # to be set on runtime

    # wandb params
    use_wandb = True
    wandb_project = 'wavenet'

    # data location
    data_root = 'ljspeech/wavs/'
    metadata_file = 'ljspeech/metadata.csv'

    # checkpoints
    checkpoint_dir = 'checkpoints/'
    checkpoint_template = 'checkpoints/wavenet{}.pt'
    model_checkpoint = 'checkpoints/wavenet1.pt'
    load_model = False

    # data processing
    valid_ratio = 0.05
    audio_length = 16384

    # melspectogramer params
    sample_rate = 22050
    win_length = 1024
    hop_length = 256
    n_fft = 1024
    f_min = 0
    f_max = 8000
    num_mels = 80
    power = 1.0
    pad_value = 1e-5
    mu = 256

    # WaveNet params
    upsample_kernel = 800
    residual_channels = 120
    skip_channels = 240
    causal_kernel = 2
    num_blocks = 16
    dilation_cycle = 8

    # optimizer params
    lr = 3e-4
    weight_decay = 0.0

    # training params
    start_epoch = 1
    num_epochs = 10
    log_steps = 500
    batch_size = 6


def set_params():
    return Params()


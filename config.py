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
    data_root = '../../DB/electronic_music_clean/'
    metadata_file = 'ljspeech/metadata.csv'

    # checkpoint params
    checkpoint_dir = 'checkpoints/'
    checkpoint_template = 'checkpoints/wavenet{}.pt'
    model_checkpoint = 'checkpoints/wavenet7.pt'
    load_model = False

    # example params
    example_spectrogram = 'example/spectrogram.pt'
    example_audio = 'example/generated.wav'
    ground_truth_audio = 'example/ground_truth.wav'
    music_audio = 'example/music.wav'
    generated_audio = 'example/generated_music.wav'
    generation_length = 16000

    # data processing
    valid_ratio = 0.05
    audio_length = 15360

    # MelSpectrogram params
    sample_rate = 8000
    win_length = 1024
    hop_length = 256
    n_fft = 1024
    f_min = 0
    f_max = 8000
    num_mels = 80
    power = 1.0
    pad_value = 1e-5
    mu = 256  # MuLawQuantization parameter

    # WaveNet params
    upsample_kernel = 800
    residual_channels = 120
    skip_channels = 240
    causal_kernel = 2
    num_blocks = 20
    dilation_cycle = 10

    # optimizer params
    lr = 3e-4
    weight_decay = 0.0

    # training params
    start_epoch = 1
    num_epochs = 50
    log_steps = 100
    batch_size = 6


def set_params():
    return Params()


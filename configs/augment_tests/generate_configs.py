import os

noise_types = ["none", "gaussian", "dropout", "shuffle",'all']
noise_modalities = ["text", "audio", "visual", "all"]
noise_levels = {}
noise_levels['low'] = [0.0, 0.01, 0.15, 0.15, [0.01, 0.15, 0.15]]  # Fourth element is a list
noise_levels['high'] = [0.0, 0.1, 0.4, 0.4, [0.1, 0.4, 0.4]]

# Train-Test
# i=0
# for train_mod in noise_modalities:
#     for test_mod in noise_modalities:
#         for train_level in ['high', 'low']:
#             for test_level in ['high', 'low']:
#                 for train_type in noise_types:
#                     for test_type in noise_types:
#                         # if we train with noise, we only want to test with the same noise and all types of noises
#                         if (train_type != 'none' ) and (train_type != test_type) and (test_type != 'all'):
#                             continue
#                         print(f"{i}  train {train_mod} {train_type} {train_level} | test {test_mod} {test_level} {test_type}")
#                         i+=1

# Train only
i=0
for train_mod in noise_modalities:
    for train_level in ['high', 'low']:
        for j, train_type in enumerate(noise_types):
            print(f"{i}  train {train_mod} {train_type} {train_level} ({noise_levels[train_level][j]})")
            i+=1
            filename = f"{train_mod}_{train_type}_{train_level}.yaml"
            with open(filename, 'w') as config_out:
                config_out.write(
                f"""
                data_dir: /content/drive/MyDrive/CMU-MOSI
                results_dir: /content/drive/MyDrive/our_results
                train: true
                test: true
                overfit_batch: false
                debug: false
                device: cuda
                num_classes: 1

                model:
                    text_input_size: 300
                    audio_input_size: 74
                    visual_input_size: 35
                    projection_size: 100
                    text_layers: 1
                    audio_layers: 1
                    visual_layers: 1
                    bidirectional: true
                    encoder_type: lstm
                    dropout: !!float 0.2
                    attention: true
                    feedback: true
                    feedback_type: learnable_sequence_mask
                    mask_index_train: 1
                    mask_index_test: 1
                    mask_dropout_train: 0.0
                    mask_dropout_test: 0.0
                    noise_type: {train_type}
                    noise_percentage_train: {noise_levels[train_level][j]}
                    noise_percentage_test: 0.0
                    noise_modality: {train_mod}
                    enable_plot_embeddings: true

                experiment:
                    name: noise_{train_mod}_{train_type}_{train_level}
                    description: MOSEI sentiment task

                dataloaders:
                    batch_size: 32
                    num_workers: 1
                    pin_memory: false

                optimizer:
                    name: Adam
                    learning_rate: !!float 5e-4

                trainer:
                    patience: 10
                    max_epochs: 100
                    retain_graph: true
                    load_model: mosei-sentiment-audio-text-visual_checkpoint.best.pth
                    checkpoint_dir: /home/geopar/projects/mm/clean_code/checkpoints
                """
                )

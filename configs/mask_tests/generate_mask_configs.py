import os

mask_index = range(1, 6)

i=0
for idx in mask_index:
    print(f"{i}  train-test-{idx}")
    i+=1
    filename = f"mask_train-test-{idx}.yaml"
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
            mask_index_train: {idx}
            mask_index_test: {idx}
            mask_dropout_train: 0.0
            mask_dropout_test: 0.0
            noise_type: none
            noise_percentage_train: 0.0
            noise_percentage_test: 0.0
            noise_modality: all
            enable_plot_embeddings: true

        experiment:
            name: mask_train-test-{idx}
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

import yaml
import argparse
import logging
from rfdetr import RFDETRMedium, RFDETRLarge, RFDETRSmall, RFDETRNano, RFDETRBase
import logging
import os
from datetime import date

def setup_argparser():
    """Set up the argument parser for training configuration.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Train RF-DETR-LMI Object Detector")
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        required=True,
        help='Path to the YAML configuration file.'
    )
    return parser


def load_config(config_path):
    """Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file. 

    Returns:
        dict: Configuration parameters.
    """ 
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_config(config):
    """Parse configuration parameters.

    Args:
        config (dict): Configuration parameters.

    Returns:
        dict: Parsed training parameters.
    """
    model_configs = {
        'model_type': config.get('model_type', 'medium'),
        'task_type': config.get('task_type', 'od')
    }
    training_configs = config.get('training', {})

    if training_configs == {}:
        raise ValueError("Training configuration is missing.")

    return {
        'model_configs': model_configs,
        'training_configs': training_configs
    }


def intiate_training(configs):
    """
    Docstring for intiate_training
    
    :param configs: Description
    """
    model_type = configs['model_configs'].get('model_type', 'medium')
    
    if model_type == 'nano':
        model = RFDETRNano()
    elif model_type == 'small':
        model = RFDETRSmall()
    elif model_type == 'medium':
        model = RFDETRMedium()
    elif model_type == 'base':
        model = RFDETRBase()
    elif model_type == 'large':
        model = RFDETRLarge()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    training_params = configs['training_configs']
    # update the output directory to have todays date
    version = 1
    output_dir = training_params.get('output_dir')
    if output_dir is None:
        raise ValueError("output_dir must be specified in training configuration.")
    output_dir  = os.path.join(output_dir, f"{date.today()}-v{version}")
    while os.path.exists(output_dir):
        version += 1
        output_dir = os.path.join(output_dir, f"{date.today()}-v{version}")
    training_params['output_dir'] = output_dir
    model.train(**training_params)
    return model



def main():
    parser = setup_argparser()
    args = parser.parse_args()

    config = load_config(args.config)
    configs = parse_config(config)
    intiate_training(configs)

if __name__ == "__main__":
    main()
import yaml
import argparse
import logging
from rfdetr import RFDETRMedium, RFDETRLarge, RFDETRSmall, RFDETRNano, RFDETRBase
import logging
import os
from datetime import date
from rf_detr_lmi.convert import convert_to_tensorrt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        'operation': config.get('operation', 'train'),
    }
    training_configs = config.get('training', {})
    conversion_configs = config.get('conversion', {})

    if training_configs == {} and model_configs['operation'] == 'train':
        raise ValueError("Training configuration is missing.")
    
    elif model_configs['operation'] == 'convert':
        if conversion_configs == {}:
            raise ValueError("Conversion configuration is missing.")
    else:
        raise ValueError(f"Unsupported operation: {model_configs['operation']}")

    return {
        'model_configs': model_configs,
        'training_configs': training_configs,
        'conversion_configs': conversion_configs    
    }

def load_model(configs):
    """
    Load the RF-DETR model based on the configuration.

    Args:
        configs (dict): Configuration parameters.
    Returns:
        RFDETR Model: An instance of the RF-DETR model.
    """
    model_type = configs['model_configs'].get('model_type', 'medium')
    operation = configs['model_configs'].get('operation', 'train')
    if operation == 'convert' and configs.get('conversion_configs') is {}:
        raise ValueError("Conversion configuration is missing.")
    if model_type == 'nano':
        if operation == 'train':
            return RFDETRNano()
        else:
            conversion_configs = configs.get('conversion_configs', {})
            return RFDETRNano(**conversion_configs)

    elif model_type == 'small':
        if operation == 'train':
            model = RFDETRSmall()
        else:
            conversion_configs = configs.get('conversion_configs', {})
            model = RFDETRSmall(**conversion_configs)
        return model
    elif model_type == 'medium':
        if operation == 'train':
            model = RFDETRMedium()
        else:           
            conversion_configs = configs.get('conversion_configs', {})
            model = RFDETRMedium(**conversion_configs)
        return model
    elif model_type == 'base':
        if operation == 'train':
            model = RFDETRBase()
        else:
            conversion_configs = configs.get('conversion_configs', {})
            model = RFDETRBase(**conversion_configs)
        return model
    elif model_type == 'large':
        if operation == 'train':
            model = RFDETRLarge()
        else:
            conversion_configs = configs.get('conversion_configs', {})
            model = RFDETRLarge(**conversion_configs)
            print(**conversion_configs)
        return model
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
def initiate_training(configs):
    """
    Docstring for initiate_training
    
    :param configs: Description
    """
    model = load_model(configs)
    
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
    if configs.get('model_configs').get('operation') == 'train':
        initiate_training(configs)
    if configs.get('model_configs').get('operation') == 'convert':
        logger.info("Starting model conversion to ONNX format...")
        # model = load_model(configs)
        output_dir = configs.get('conversion_configs', {}).get('output_dir', os.path.dirname(configs.get('conversion_configs', {}).get('pretrain_weights', '')))
        if os.path.isfile(os.path.join(output_dir, 'inference_model.onnx')):
            logger.info(f"ONNX model already exists at {os.path.join(output_dir, 'inference_model.onnx')}. Skipping export.")
        else:
            model = load_model(configs)
            model.export(output_dir=output_dir) # export to ONNX
        
        logger.info("Converting to TensorRT engine...")
        convert_to_tensorrt(os.path.join(output_dir, 'inference_model.onnx'))
        
        


if __name__ == "__main__":
    main()
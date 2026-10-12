from ml_model import train_models
if __name__ == '__main__':
    accuracy=train_models()
    print(f'Demo model trained. Holdout accuracy on synthetic data: {accuracy:.1%}')
    print('Important: this is synthetic demonstration data, not proof of real-world accuracy.')

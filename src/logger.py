import wandb


class WanDBWriter:
    def __init__(self, project='hifi_project'):
        wandb.login(key='777734be0649971345886f08a6b84c9b9b190223')
        wandb.init(project=project)

    def add_metrics(self, metrics):
        wandb.log(metrics)

    def add_audio(self, pred, true, transcript):
        wandb.log({
            'pred audio': wandb.Audio(pred.squeeze().numpy(), sample_rate=22050, caption=transcript),
            'true audio': wandb.Audio(true.squeeze().numpy(), sample_rate=22050, caption=transcript)
        })

    def add_spectrogram(self, pred, true, transcript):
        wandb.log({
            'pred spectrogram': wandb.Image(pred.squeeze().numpy(), caption=transcript),
            'true spectrogram': wandb.Image(true.squeeze().numpy(), caption=transcript)
        })

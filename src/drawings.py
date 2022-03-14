from pathlib import Path
from typing import Union

import numpy as np
import seaborn as sns
import tikzplotlib
import torch
from matplotlib import pyplot as plt

from utils import stats_from


def draw_violins(x: torch.Tensor, x_hat: torch.Tensor, savefig: Union[Path, str] = None, savetikz: bool = False):
    """
    Draws a violin plot

    Parameters
    ----------
    x: torch.Tensor
        Original data
    x_hat: torch.Tensor
        Sampled values
    savefig: Path, str
        Output path to save figure. If None, shows the figure
    savetikz: bool
        Save tikz file
    """
    with plt.style.context(['science', 'ieee', 'grid']):
        plt.figure(dpi=256)
    
        ax = sns.violinplot(data=[x, x_hat], palette={0: "tab:red", 1: "tab:blue"})
        ax.set_xticklabels(["Observed", "Best\ individual"])
    
        # TODO: Añadir o corregir el título
        # ax.set_title(f'$HoF ({self.hof[0].fitness.x_hat[0]:.4f})\ vs.\ observed$')
        ax.grid(True)
    
        plt.tight_layout()
        if savetikz and savefig is not None:
            tikzfile = savefig.parent / savefig.name.replace(".png", ".tex")
            phi = 1.618033988
            scale = 1
            tikzplotlib.save(tikzfile.as_posix(),
                             # extra_axis_parameters=["dashed"],
                             axis_width=f'{scale}\linewidth',
                             axis_height=f'{scale / phi:.3}\linewidth')
    
        if savefig is not None:
            plt.savefig(savefig)
        else:
            plt.show()
    
        plt.close()


def _is_discrete_array(x: torch.Tensor) -> bool:
    """
    Check if an array is formed by integers

    Parameters
    ----------
    x: torch.Tensor

    Returns
    -------
    bool

    Examples
    --------
    >>> x = torch.Tensor([5, 2, 6, 1, 6, 9, 5])
    >>> _is_discrete_array(x)
    True
    >>> _is_discrete_array(x + 0.233)
    False
    """
    idx = (x > 0.5) | (x < -0.5)

    return bool(torch.sum(x[idx] % x[idx].round()) == 0)


def _hist(x: torch.Tensor, x_hat: torch.Tensor = None, type: str = "bar", log: bool = False):
    x_, y_ = torch.unique(x, return_counts=True)
    offset = 0

    if _is_discrete_array(x):
        width = 0.4
        if x_hat is not None:
            offset = width / 2

        # Observed values are integers
        plt.bar(x_ - offset, y_ / y_.sum(), width=width, label="Observed", color="tab:red")

        if x_hat is not None:
            # Sampled values
            x_hat_, y_hat_ = torch.unique(x_hat.round(), return_counts=True)
            plt.bar(x_hat_ + offset, y_hat_ / y_hat_.sum(), width=width, label="Sampled", color="tab:blue")
    else:
        # Observed values are floats
        bins = 50

        a = x.min()
        b = x.max()
        if x_hat is not None:
            a = min(a, x_hat.min())
            b = max(b, x_hat.max())

        dist = abs(a - b)
        if dist < 5:
            digits = 3
        elif dist < 15:
            digits = 2
        elif dist < 50:
            digits = 1
        else:
            digits = 0

        width = dist / 150
        if x_hat is not None:
            offset = width / 2

        x = torch.from_numpy(x.numpy().round(digits))
        x_ = torch.from_numpy(torch.linspace(a, b, bins).numpy().round(digits))
        y_ = torch.histc(x, bins=bins, min=a, max=b)

        if log:
            y_[y_ > 0] = torch.log10(y_[y_ > 0])

        if type == "line":
            plt.plot(x_ - offset, y_ / y_.sum(), label="Observed", color="tab:red", lw=0.5)
        else:
            plt.bar(x_ - offset, y_ / y_.sum(), width=width, label="Observed", color="tab:red")

        if x_hat is not None:
            # Sampled values
            x_hat = torch.from_numpy(x_hat.numpy().round(digits))
            y_hat_ = torch.histc(x_hat, bins=bins, min=a, max=b)
            if log:
                y_hat_[y_hat_ > 0] = torch.log10(y_hat_[y_hat_ > 0])

            if _is_discrete_array(x_hat):
                plt.bar(x_ + offset, y_hat_ / y_hat_.sum(), width=width, label="Sampled", color="tab:blue")
            else:
                plt.plot(x_ + offset, y_hat_ / y_hat_.sum(), label="Sampled", color="tab:blue", linewidth=0.5)

    ylabel = "Frequency"
    if log:
        ylabel = "Frequency $(\log_{10})$"
    plt.ylabel(ylabel)


def draw_histogram(x: torch.Tensor,
                   x_hat: torch.Tensor = None,
                   savefig: Union[Path, str] = None,
                   type: str = "bar",
                   log: bool = False,
                   savetikz: bool = False):
    """
    Draws histogram depending on the type of the data. If data is integer,
    draws each point with a bar plot. If data is float, draws bins depending
    on the number of uniques rounded values

    Parameters
    ----------
    x: torch.Tensor
        Original data
    x_hat: torch.Tensor
        Sampled values
    savefig: Path, str
        Output path to save figure. If None, shows the figure
    type: str
        'bar' for bar plot. 'line' for line plot
    log: bool
        Applies log10 transformation to data
    savetikz: bool
        Save tikz file
    """
    with plt.style.context(['science', 'ieee', 'grid']):
        plt.figure(dpi=256)

        _hist(x, x_hat=x_hat, type=type, log=log)

        # ax.text(ax.get_xticks()[1],  # - abs(ax.get_xticks()[1] - ax.get_xticks()[0]) * .2,
        #         ax.get_yticks()[-2] * .97,
        #         r'\begin{tabular}{lrrrr}'
        #         # r'\toprule '
        #         r'{} & $mean$ & $var$ & $skew$ & $kurt$\\'
        #         r'\midrule '
        #         r'$Observed$ &' + " & ".join(map(lambda x: f"${x:.5f}$",
        #                                          stats_df.loc[
        #                                              "Observed"].x_hat)) + r' \\$Best\ ind.$ &' + " & ".join(
        #             map(lambda x: f"${x:.5f}$", stats_df.loc["Drawn"].x_hat)) + r' \\'
        #         # r'\bottomrule'
        #                                                                          r'\end{tabular}',
        #         fontsize=9
        #         )
        #

        # plt.yticks(torch.linspace(0, 1, 11))
        plt.legend(prop={'size': 8})

        plt.tight_layout()
        if savetikz and savefig is not None:
            tikzfile = savefig.parent / savefig.name.replace(".png", ".tex")
            phi = 1.618033988
            scale = 1
            tikzplotlib.save(tikzfile.as_posix(),
                             # extra_axis_parameters=["dashed"],
                             axis_width=f'{scale}\linewidth',
                             axis_height=f'{scale / phi:.3}\linewidth')

        if savefig is not None:
            plt.savefig(savefig, dpi=256)
        else:
            plt.show()
        plt.close()


def draw_stats(x: torch.Tensor, x_hat: torch.Tensor, savefig: Union[Path, str] = None, savetikz: bool = False):
    """
    Draws stats

    Parameters
    ----------
    x: torch.Tensor
        Original data
    x_hat: torch.Tensor
        Sampled values
    savefig: Path, str
        Output path to save figure. If None, shows the figure
    savetikz: bool
        Save tikz file
    """
    # Compare data stats in the same scale
    x_ = stats_from(x)
    x_hat_ = stats_from(x_hat)
    # xlabs = ["mean", "variance", "skewness", "kurtoses"]
    xlabs = ["$\mu_1$", "$\mu_2$", "$\mu_3$", "$\mu_4$"]

    xlocs = torch.arange(4)
    width = 0.4
    offset = width / 2
    with plt.style.context(['science', 'ieee', 'grid']):
        plt.figure(dpi=256)
    
        plt.bar(xlocs - offset, x_, width=width, label="Observed", color="tab:red")
        plt.bar(xlocs + offset, x_hat_, width=width, label="Sampled", color="tab:blue")

        for i in xlocs:
            plt.text(xlocs[i] - width * .5,
                     x_[i],
                     round(x_[i].item(), 2),
                     ha="center",
                     va="bottom",
                     fontsize=6)
            plt.text(xlocs[i] + width * .5,
                     x_hat_[i],
                     round(x_hat_[i].item(), 2),
                     ha="center",
                     va="bottom",
                     fontsize=6)

        plt.legend()
        plt.xticks(xlocs, xlabs)
        plt.tight_layout()

        if savetikz and savefig is not None:
            tikzfile = savefig.parent / savefig.name.replace(".png", ".tex")
            phi = 1.618033988
            scale = 1
            tikzplotlib.save(tikzfile.as_posix(),
                             # extra_axis_parameters=["dashed"],
                             axis_width=f'{scale}\linewidth',
                             axis_height=f'{scale / phi:.3}\linewidth')

        if savefig is not None:
            plt.savefig(savefig)
        else:
            plt.show()
        plt.close()


def compare(X: torch.Tensor,
            model: torch.distributions.Distribution,
            working_dir: Path,
            id_: str,
            save_prefix: str,
            scaler=None,
            n_samples: int = 5000,
            savetikz: bool = False):
    # Draw values from the best evolved individual
    x_hat = model.sample([n_samples]).ravel()
    
    # If any type of scaling have been applied, then apply de inverse
    # transformation to get the original scaled values
    if scaler:
        x_hat = torch.from_numpy(scaler.inverse_transform(x_hat.reshape(-1, 1))).ravel()
    
    working_dir.mkdir(parents=True, exist_ok=True)
    
    # Draw stats
    draw_stats(X, x_hat, savefig=working_dir / f"{id_}_{save_prefix}stats.png", savetikz=savetikz)
    
    # Draw violin plots
    draw_violins(X, x_hat, savefig=working_dir / f"{id_}_{save_prefix}comparison_violin.png", savetikz=savetikz)
    
    # Draw histogram
    draw_histogram(X, x_hat, savefig=working_dir / f"{id_}_{save_prefix}comparison.png", savetikz=savetikz)


def plot_data(x: torch.Tensor,
              x_hat: torch.Tensor = None,
              xlabel: str = "",
              outfile: str = "",
              savetikz: bool = True,
              **kwargs):
    """
    Plot data histogram included in memory

    Parameters
    ----------
    savetikz: bool
        Save tikz file
    x: torch.Tensor
        Data array
    xlabel: str
    outfile: str
        Output file path
    **kwargs
        Arguments for _hist() function
    """
    
    outdir = Path(__file__).parents[1] / "memoria"
    
    # Draw histogram
    with plt.style.context(['science', 'ieee', 'grid']):
        plt.figure(figsize=(19.8 / 4, 10.8 / 4))
        log = kwargs["log"] if "log" in kwargs.keys() else False
        type_ = kwargs["type"] if "type" in kwargs.keys() else "bar"
        _hist(x, x_hat=x_hat, type=type_, log=log)

        plt.grid(True, alpha=0.2)

        if xlabel:
            plt.xlabel(xlabel)

        plt.tight_layout()

        if savetikz and outfile != "":
            tikzfile = outdir / outfile.replace(".png", ".tex")
            phi = 1.618033988
            scale = 1
            tikzplotlib.save(tikzfile.as_posix(),
                             # extra_axis_parameters=["dashed"],
                             axis_width=f'{scale}\linewidth',
                             axis_height=f'{scale / phi:.3}\linewidth')

        if outfile:
            plt.savefig(outdir / "images" / outfile, dpi=256)
        else:
            plt.show()
    plt.close()


def plot_elbo(losses, savefig: Union[Path, str] = None):
    with plt.style.context(['science', 'ieee', 'grid']):
        plt.figure(figsize=(19.8 / 4, 10.8 / 4))
        losses[losses > losses.quantile(.75)] = np.nan  # Remove high values
        plt.plot(losses, label="ELBO")
        plt.xlabel("step")
        plt.ylabel("loss")
        # plt.yscale("log")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if savefig:
            plt.savefig(savefig, dpi=256)
        else:
            plt.show()
        plt.close()


def plot_latent_values(latent_values: torch.Tensor, _args: list, savefig: Union[Path, str] = None):
    with plt.style.context(['science', 'ieee', 'grid']):
        fig, axs = plt.subplots(len(_args), 1, figsize=(19.8 / 2, 10.8 / 2))
        for i, ax in enumerate(axs):
            v = latent_values[:, i].detach().numpy()
            ax.plot(v, label="Learned")
            ax.grid(True)
            ax.set_ylabel(["x", "y", "z"][i])

            # Plot value used
            ax.hlines(_args[i], 0, len(v), color="g", linestyles="--", label="Original")
            ax.legend()

        plt.suptitle(f"Latent values")
        plt.xlabel("step")
        plt.tight_layout()
        if savefig:
            plt.savefig(savefig, dpi=128)
        else:
            plt.show()
        plt.close()


def ssim_example():
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import mean_squared_error
    from sklearn.datasets import load_digits

    mnist = load_digits()
    img = mnist.images[1] / 255
    rows, cols = img.shape

    noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
    noise[np.random.random(size=noise.shape) > 0.5] *= -1

    img_noise = img + 2 * noise
    img_const = img + abs(noise)

    mse_none = mean_squared_error(img, img)
    ssim_none = ssim(img, img, data_range=img.max() - img.min())

    mse_noise = mean_squared_error(img, img_noise)
    ssim_noise = ssim(img, img_noise, data_range=img_noise.max() - img_noise.min())

    mse_const = mean_squared_error(img, img_const)
    ssim_const = ssim(img, img_const, data_range=img_const.max() - img_const.min())

    label = 'MSE: {}, SSIM: {:.2f}'

    with plt.style.context(["ieee", "science"]):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
        ax = axes.ravel()
        ax[0].imshow(img, cmap="gray",
                     vmin=0,
                     # vmax=1
                     )
        ax[0].set_xlabel(label.format(mse_none, ssim_none))
        # ax[0].set_title('Original image')
        ax[0].grid(False)
        ax[0].axis("off")
        print(label.format(mse_none, ssim_none))

        ax[1].imshow(img_noise, cmap="gray",
                     vmin=0,
                     # vmax=1
                     )
        ax[1].set_xlabel(label.format(mse_noise, ssim_noise))
        # ax[1].set_title('Image with noise')
        ax[1].grid(False)
        ax[1].axis("off")
        print(label.format(mse_noise, ssim_noise))

        ax[2].imshow(img_const, cmap="gray",
                     vmin=0,
                     # vmax=1
                     )
        ax[2].set_xlabel(label.format(mse_const, ssim_const))
        # ax[2].set_title('Image plus constant')
        ax[2].grid(False)
        ax[2].axis("off")
        print(label.format(mse_const, ssim_const))

        plt.tight_layout()
        plt.savefig("../memoria/images/ssim.png", dpi=256)
        # plt.show()
        plt.close()


if __name__ == "__main__":
    from utils import load_temperature, load_reaction_times, load_precipitation, load_normal, load_beta
    
    x = load_temperature()
    # x_hat = torch.distributions.Normal(x.mean(), x.std()).sample([500])
    # draw_violins(x, x_hat)
    # draw_histogram(x, x_hat)
    plot_data(x, xlabel='Temperature (ºC)', savetikz=True, outfile="temperature.png")
    # draw_stats(x, x_hat)

    x = load_reaction_times(6)
    # x_hat = torch.distributions.Normal(x.mean(), x.std()).sample([500])
    # draw_violins(x, x_hat)
    # draw_histogram(x, x_hat)
    plot_data(x, xlabel='Reaction time (ms)', savetikz=True, outfile="reaction_times.png")
    # draw_stats(x, x_hat)

    x = load_precipitation()
    # draw_violins(x, x)
    # draw_histogram(x, log=True)
    plot_data(x, xlabel="mm", savetikz=True, log=True, type="bar", outfile="precipitation_log.png")
    plot_data(x, xlabel="mm", savetikz=True, log=False, type="bar", outfile="precipitation.png")
    
    plot_data(load_normal(), savetikz=True, type="bar", outfile="normal.png")
    plot_data(load_beta(), savetikz=True, type="bar", outfile="beta.png")
    
    ssim_example()

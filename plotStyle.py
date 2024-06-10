# Definition of custom plot style
import matplotlib.font_manager as fm

# Import new font for plots
fm.FontManager.addfont(fm.fontManager, path="/scr/jpember/polly/Quicksand-Regular.ttf")

lw = 1.3

plotStyle = {
    # Set font to the beautiful Quicksand:
    # https://github.com/andrew-paglinawan/QuicksandFamily
    "font.family" : "Quicksand",
    # Also try to use this font for mathtext characters
    "mathtext.default" : "regular",
    
    # Set axis text label sizes
    "axes.labelsize" : 12,
    "axes.titlesize" : 14,
    # Axis spine line widths
    "axes.linewidth" : lw,
    # Optional: set the default axis limits to be round numbers
    "axes.autolimit_mode" : "round_numbers",
    
    # x tick properties
    "xtick.top" : True,
    "xtick.bottom" : True,
    "xtick.major.size" : 8,
    "xtick.major.width" : lw,
    "xtick.minor.visible": True,
    "xtick.minor.size" : 4,
    "xtick.minor.width" : lw,
    "xtick.direction" : "in",
    "xtick.major.pad" : 10,
    
    # y tick properties
    "ytick.left" : True,
    "ytick.right" : True,
    "ytick.major.size" : 8,
    "ytick.major.width" : lw,
    "ytick.minor.visible": True,
    "ytick.minor.size" : 4,
    "ytick.minor.width" : lw,
    "ytick.direction" : "in",
    "ytick.major.pad" : 10,
    
    # Tick text label sizes
    "xtick.labelsize" : 12,
    "ytick.labelsize" : 12,
    
    # Legend properties
    "legend.frameon" : False,
    "legend.fontsize" : 14,
    "legend.labelspacing" : 0.25,
    "legend.handletextpad" : 0.25,
    
    # Default figure size and constrined_layout (previously plt.tight_layout())
    "figure.figsize" : (14 / 2.54, 10 / 2.54),
    "figure.dpi" : 96,
    "figure.constrained_layout.use" : True,
    
    # Default properties for plotting lines, markers, scatterpoints, errorbars, etc.
    "lines.linewidth" : 2,
    "lines.markeredgecolor" : "k",
    "lines.markersize" : 16,
    "lines.solid_capstyle" : "round",
    "lines.dash_capstyle" : "round",
    "scatter.edgecolors" : "k",
    "errorbar.capsize" : 3,
    "hist.bins" : 20,
}

# This file can be sourceed to activate tab completion for asap commands.
# It may be placed at $CONDA_PREFIX/etc/conda/activate.d/ to enable the compeletion
# when the environment is activated
if [ ! -z "$BASH_VERSION" ]; then
	eval "$(_ASAP_COMPLETE=source_bash asap)"
elif [ ! -z "$ZSH_VERSION" ]; then
	eval "$(_ASAP_COMPLETE=source_zsh asap)"
fi

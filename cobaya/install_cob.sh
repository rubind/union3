conda create -n cobaya_w0wa python=3.11 -y
conda activate cobaya_w0wa
conda install scipy numpy matplotlib
python -m pip install --upgrade pip
python -m pip install cobaya
mkdir -p $HOME/cobaya_packages
cobaya-install desi_dr2_bao_planck_pantheonplus_w0wa.yaml --packages-path $HOME/cobaya_packages

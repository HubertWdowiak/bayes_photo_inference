#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J NN_Train
## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=1
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 4GB na rdzeń)
#SBATCH --mem-per-cpu=32GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=15:00:00
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgdyplomanci3
## Specyfikacja partycji
#SBATCH --partition plgrid
## Plik ze standardowym wyjściem
#SBATCH --output="/net/people/plgmazurekagh/CyfroVet/ML_project/Code/output_files/output.out"
## Plik ze standardowym wyjściem błędó0w
#SBATCH --error="/net/people/plgmazurekagh/CyfroVet/ML_project/Code/error_files/error.err"
## przejscie do katalogu z ktorego wywolany zostal sbatch
cd /net/people/plgmazurekagh/MLProject/newvenv/bin/
module add plgrid/tools/python/3.9
source /net/people/plgmazurekagh/MLProject/newvenv/bin/activate
cd /net/people/plgmazurekagh/CyfroVet/ML_project/Code/Python_Code
##pip install -r /net/people/plgmazurekagh/requirements.txt
python /net/people/plgmazurekagh/CyfroVet/ML_project/Code/Python_Code/ML_Veterinary_NeuralNetworks.py
##deactivate # decativate venv
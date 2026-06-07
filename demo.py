import subprocess
import sys

subprocess.run([sys.executable,'./Seq2Seq/trans.py'])
subprocess.run([sys.executable,'./Seq2Seq/model.py'])
subprocess.run([sys.executable,'./Seq2Seq/translation.py'])
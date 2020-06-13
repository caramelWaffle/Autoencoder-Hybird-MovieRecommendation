from subprocess import check_output

command = '/home/jzliu/localperl/bin/perl /home/jzliu/Multilingual_PreSumm/ROUGE-1.5.5/ROUGE-1.5.5.pl -e /home/jzliu/Multilingual_PreSumm/ROUGE-1.5.5/data -c 95 -m -r 1000 -n 2 -a /home/jzliu/Multilingual_PreSumm/temp/tmp8cp77c78/rouge_conf.xml'
rouge_output = check_output(command).decode("UTF-8")
print(rouge_output)
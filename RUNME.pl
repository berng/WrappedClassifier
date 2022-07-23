#!/usr/bin/perl
@models=(
	'model.classifier.withHeff.noReg.gm20.iri2012.ec', 
	);

$f='./tmp.tmp';
 if(-s $f >0)
 {
  print "f: $f SD: $subdir FN: $fname;\n";
  foreach $model (@models)
  {
   $mod_int=$model;
   $mod_int=~ s/\/model\.classifier//g;
   $mod_int=~ s/\//\_/g;
   print "model: $mod_int;\n";
   `mkdir -p out/$mod_int/$subdir`;
   `python apply_model_exact_format_withheff-newmodel.py $f $model out.tmp`;
  }
 }

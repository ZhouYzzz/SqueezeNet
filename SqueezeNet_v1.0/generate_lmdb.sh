TOOLS='/home/gpu/zhouyz/caffe/build/tools'
rm -r lmdb_cuhk_test lmdb_cuhk_train
$TOOLS/convert_imageset ./ test.txt lmdb_cuhk_test
echo DONE

$TOOLS/convert_imageset ./ train.txt lmdb_cuhk_train
echo DONE
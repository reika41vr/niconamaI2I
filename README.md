# niconamaImg2Img
ニコ生のコメントをトリガーにStable Diffusionでimg2imgするツール


## 概要
以下のコマンドで実行します。必要なライブラリは適宜入れてください。Diffusersなどが必要です。  
python niconamai2i.py  
  
  
実行中は、ニコ生の"i2i "から始まるコメントに反応して画像がImg2Imgで変化します。  
変化前の画像はimage1.pngという名前で同じフォルダに置いてください。  
変化後の画像はimage1.pngに上書きされるほか、image2.png,image3.pngというファイルにも出力(上書き)されます。  
これは画像が更新された際にOBSでの表示が消えることがあり、別々のファイルをOBS上で重ねて配置しておくことで(見かけ上は)表示が消えにくくなるためです。  
また、実行状態はi2imsg.txtというファイルに保存されます。これもOBSでの表示用です。  

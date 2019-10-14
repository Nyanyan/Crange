# Crange
Change cross color of speed cubing video

**日本語は下部にあります。**

![GitHub last commit](https://img.shields.io/github/last-commit/Nyanyan/Crange)[![GitHub license](https://img.shields.io/github/license/Nyanyan/Crange)](https://github.com/Nyanyan/Crange/blob/master/LICENSE)[![GitHub stars](https://img.shields.io/github/stars/Nyanyan/Crange)](https://github.com/Nyanyan/Crange/stargazers)![GitHub top language](https://img.shields.io/github/languages/top/Nyanyan/Crange)![GitHub language count](https://img.shields.io/github/languages/count/Nyanyan/Crange)![GitHub repo size](https://img.shields.io/github/repo-size/Nyanyan/Crange)![GitHub followers](https://img.shields.io/github/followers/Nyanyan?style=social)![Twitter Follow](https://img.shields.io/twitter/follow/Nyanyan_Cube?style=social)

## Abstract

Are you a white-crossed cuber, or a blue-crossed cuber(, or color neutral)?

You may feel confused if you see a video in which different-color-crossed cuber do something. This software, Crange, changes the cross color of that kind of video. Crange is named after “Change” and “Cross”.

## How to Install

**This software supports only Windows 32 & 64 bit, Tested Windows10 64bit only**

### Through zip file

1. Visit https://1drv.ms/u/s!AlopFnI_9zPsjtd4CCQTdMzp79JZRQ?e=gxFNgz and download the latest version

2. Unzip it 

3. Execute “crangeX.exe”

### Through GitHub

1. Clone repository
2. Execute “archive/crangeX/crangeX.exe”

## How to Use

1. Open CrangeX.exe
2. Confirm “Input Path” and “Output Path”
   **Both path must be right paths**
   <img src="https://github.com/Nyanyan/Crange/blob/master/img/crange_1.png" width="300">
   
3. Press “Input Video” button
4. Adjust processing values
   <img src="https://github.com/Nyanyan/Crange/blob/master/img/crange_2.png" width="300">

   * Compression
     The value of compressing masks, whose colors changes (The inputted video never be compressed)
   * Lightness
     The value of light, if lighter, the colors sensed will be lighter
   * Hue
     The value of hue, if bigger, the sensed hue will be bigger
   * Deleting Size
     Crange deletes big and small masks, and this parameter is a threshold of area, which decide which area to delete. Areas bigger than the threshold and areas 1 / 10 smaller than the threshold will be deleted.
   * Deleting Mode
     Change deleting mode, delete some big and small masks or not
   * Color Mode
     Change color mode, white cross -> blue cross or blue cross -> white cross
   * Set Default
     Set all numbers to default
   * Frame Number
     You can see what these numbers you set will work with processing any one frame. This number is the frame you process.
   * Process One Frame
     The button to process one frame
5. Check status. If you start processing, the percentage will be displayed.
6. Start (and stop) processing with following button. The window will be like the image.
   <img src="https://github.com/Nyanyan/Crange/blob/master/img/crange_3.png" width="300">

   * Start
     Start processing
   * Stop
     Stop processing, if you press this button during processing, the outputted video will be halfway and has sound until the frame you stopped.

## Release Note

### Crange 4.0

First Release

### Crange 4.1



### Crange 4.2

Fixed bug

Disable buttons during processing

### Crange 4.3

Sound encoding

Enable to choose deleting mask function

Easier to use

Speed up

### Crange 4.4

Fixed bug

### Crange 4.5

Correspond to Japanese color

## 概要

世の中には大きく分けて3種類のキューバーがいます。白クロス、青クロス、CNです。

もしあなたが白クロスだとして、見たい動画が青クロスだったとします。特にこの動画が何かの解説をする動画であった場合、その動画はこの流派のち外のためにとても見にくいものであるでしょう。この“Crange”というソフトウェアは、動画中のキューブのクロス色を変更します。“Crange”は“Change”と“Cross”から命名されました。

## インストール方法

**このソフトウェアはWindowsの32、64bit専用です。また、テスト環境はWindows10 64bitだけです。**

### Zipファイルから

1. ここ https://1drv.ms/u/s!AlopFnI_9zPsjtd4CCQTdMzp79JZRQ?e=gxFNgz から最新バージョンをダウンロードしてください。
2. 解凍してください。
3. “crangeX.exe”を実行してください。

### GitHubから

1. リポジトリをクローンしてください。
2. “archive/crangeX/crangeX.exe”を実行してください。

## 使い方

1. CrangeX.exeを実行してください。

2. “Input Path”と “Output Path”の欄を入力してください。
   **それぞれは実在するパスを入力してください**
   <img src="https://github.com/Nyanyan/Crange/blob/master/img/crange_1.png" width="300">

3. “Input Video”ボタンを押してください。

4. 処理に使う数値を設定してください。
   <img src="https://github.com/Nyanyan/Crange/blob/master/img/crange_2.png" width="300">

   - Compression
     マスク(色を変えるところ)の圧縮具合です。

   - Lightness
     光量。大きくすると認識される色は明るくなります。

   - Hue
     色相。大きくするとそれだけ色相がずれます。

   - Deleting Size

     Crangeは大きすぎるマスクは色を変更しないように設定ができ、この値はその消去する面積に関する値です。この閾値よりも大きなマスクとこの閾値の1/10よりも小さなマスクは無視されます。

   - Deleting Mode
     大きい/小さいマスクを消去するかどうかのモード変更します。

   - Color Mode
     色のモードを変更します。白クロス->青クロス、または青クロス->白クロス

   - Set Default
     全ての数値をデフォルトに戻す

   - Frame Number
     Crangeでは設定した数値が実際にどのように反映されるのか、任意の1フレームだけに処理を施して視覚的に見ることができます。これはどのフレームを処理するかを選ぶものです。

   - Process One Frame
     任意の1フレームを処理するボタン

5. ステータスを確認します。処理を開始すると処理の進み具合も表示されます。

6. 処理を開始/終了します。処理中の画面は以下のようになります。
   <img src="https://github.com/Nyanyan/Crange/blob/master/img/crange_3.png" width="300">

   - Start
     処理を開始
   - Stop
     処理を終了。このボタンを押すと、動画はボタンを押したフレームまで出力され、音声もそこまでつきます。

## リリースノート

### Crange 4.0

最初のリリース

### Crange 4.1



### Crange 4.2

バグ修正

処理中のボタン無効化

### Crange 4.3

音声エンコード

マスク消去の選択/調整機能

使いやすく

高速化

### Crange 4.4

バグ修正

### Crange 4.5

日本配色に対応
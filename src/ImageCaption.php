<!DOCTYPE html>
<html lang="ja">

<head>
    <title>画像キャプション自動生成</title>
</head>

<body>
    <?php
    $updir = "../imgs/";
    $filename = $_FILES['upfile']['name'];
    if (move_uploaded_file($_FILES['upfile']['tmp_name'], $updir . $filename) == FALSE) {
        print("Upload 失敗");
    } else { }

    exec("./model.sh ../imgs/" . $filename, $result);
    $tmp = end($result);

    $arr = explode("[", $tmp);
    ?>

    <?php
    echo '<img src="' . $updir . $filename . '">';
    echo '<p style="font-size:50px;">' . $arr[0] . '</p>';
    ?>

</body>

</html>
# モデルをS3にアップロードする
# $1 成果物へのディレクトリ

import os
import logging
import boto3
from botocore.exceptions import ClientError
from pathlib import Path

import hashlib
import shutil

def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket
    ref.
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-uploading-files.html

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    
    # 
    

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True



def upload_result(result_dir):
    
    # result_bucket = os.environment('S3_UPLOAD_RESULT_BUCKET')
    # model_bucket = os.environment('S3_UPLOAD_MODEL_BUCKET')
    
    result_bucket = 'meow-sample-result-bucket'
    
    result_dir = Path(result_dir)
    assert result_dir.exists(), result_dir
    
    config_yml = result_dir / 'config/sample.yml'
    assert config_yml.exists(), config_yml
    
    model_path = result_dir / 'model/model.pth'
    assert model_path.exists(), model_path
    
    
    # ymlのハッシュ値をファイル名として使用する。
    with config_yml.open('rb') as f:
        dirname = hashlib.md5(f.read()).hexdigest()
    print(dirname)
    
    tgt_dir = Path('/tmp') / dirname

    shutil.copytree(str(result_dir), str(tgt_dir),
                    ignore=shutil.ignore_patterns("*.pth", '*.pt'))

    # モデルは選択的に上げる
    shutil.copy(str(model_path),
                str(tgt_dir / 'model/model.pth'))
    
    shutil.make_archive(str(tgt_dir), 'zip', root_dir='/tmp', base_dir=dirname)
    # コード
    zipfile = str(tgt_dir) + '.zip'
    
    upload_file(zipfile, result_bucket, 
                object_name=Path(zipfile).name)
    



if __name__ == "__main__":
    import sys
    argvs = sys.argv

    upload_result(argvs[1])
    

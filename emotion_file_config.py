#coder: 芙宁娜_荒性
import os



class emotionGetFilePath:
    def __init__(self):
        #这里我想要获取GPT_SoVITS\emotions的具体路径,比如
        #f"GPT_SoVITS\emotions\{character}\{wavs}\{name}.wav"
        #f"GPT_SoVITS\emotions\{character}\{lists}\{name}.list"
        self.basePath = "GPT_SoVITS/emotions"
        pass
    def FileExists(self, character: str = None,type: str = None,emotion:str = None):
        return os.path.exists(f"{self.basePath}/{character}/{type}/{emotion}.{type}")

    def getFilePath(self, character: str = None,type: str = None,emotion:str = None):
        #返回前要校验文件是否存在,不存在直接返回None
        exists = os.path.exists(f"{self.basePath}/{character}/{type}/{emotion}.{type}")
        print(exists)
        if exists == False:
            return None
        else:
            return f"{self.basePath}/{character}/{type}/{emotion}.{type}"
        
    def IfNotExistsCreate(self, character: str = None, type: str = None):
        file_path = f"{self.basePath}/{character}/{type}/{type}.{type}"
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            open(file_path, 'w').close()

    def FileCreateToList(self, character: str = None, type: str = "list",emotion:str = None,text: str = None):
        file_path = f"{self.basePath}/{character}/{type}/{emotion}.{type}"
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)
                f.close()
                print(f"list写入完成")
        
emotionPath = emotionGetFilePath()

if __name__ == "__main__":
    value1 = emotionPath.getFilePath("娜维娅","wav","平静")
    value2 =emotionPath.getFilePath("娜维娅","list","平静")
    print(value1)
    print(value2)
    emotionPath.IfNotExistsCreate("芙宁娜","wav")
    emotionPath.IfNotExistsCreate("芙宁娜","list")
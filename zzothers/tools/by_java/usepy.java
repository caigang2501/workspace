import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class PythonCaller {

    public static void main(String[] args) {
        try {
            // 设置 Python 脚本路径
            String pythonScriptPath = "func4use.py";

            // 设置命令行参数
            String[] command = {"python", pythonScriptPath, "arg1_value", "arg2_value"};

            // 构建进程命令
            ProcessBuilder processBuilder = new ProcessBuilder(command);

            // 启动进程
            Process process = processBuilder.start();

            // 获取进程输出流
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;

            // 读取输出
            while ((line = reader.readLine()) != null) {
                // 输出 Python 脚本返回的值
                System.out.println("Python script returned: " + line);
            }

            // 等待进程执行完毕
            int exitCode = process.waitFor();
            System.out.println("Python script executed with exit code: " + exitCode);

        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }
}

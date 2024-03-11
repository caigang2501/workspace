import excel_tools

df = excel_tools.parse_xlsx('thirdlib/tools/2023年7月劳务派遣工资.xlsx')
print(df.columns)
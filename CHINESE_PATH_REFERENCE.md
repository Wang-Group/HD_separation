# Chinese Path Reference

This file documents repository paths whose names contain Chinese characters.

- No files or folders are renamed here.
- The translations below are short working translations for readability.
- `wx` is kept as-is because it looks like an internal project/platform prefix rather than a Chinese word.

| Path | Type | Chinese text | Short translation | Comment |
|---|---|---|---|---|
| `EC_tool_sys/manager/robotic_arm/MG400/wx_八通道电解池` | directory | `八通道电解池` | `eight-channel electrolytic cell` | MG400 robot project folder containing position/config files for the eight-channel cell setup. |
| `EC_tool_sys/manager/robotic_arm/MG400/wx_电极片推送及旋转机构` | directory | `电极片推送及旋转机构` | `electrode-sheet pushing and rotating mechanism` | MG400 robot project folder for a mechanism that pushes and rotates electrode pieces. |
| `EC_tool_sys/manager/robotic_arm/MG400/wx-8channel取出电极片` | directory | `取出电极片` | `remove electrode sheet` | MG400 robot project folder for taking electrode pieces out of the 8-channel setup. |
| `EC_tool_sys/botorch_test/test/EC_HD/GP_test/异常序号.txt` | file | `异常序号` | `abnormal indices` | Text note listing problematic or outlier sample indices for the `GP_test` experiment area. |
| `EC_tool_sys/botorch_test/test/EC_HD/.ipynb_checkpoints/异常序号-checkpoint.txt` | file | `异常序号` | `abnormal indices` | Jupyter checkpoint copy of the same note file above. |
| `EC_tool_sys/manager/configs/bopai/bopai_DESKTOP-9HA73JN_5月-13-211123-2025_Conflict.json` | file | `5月` | `May` | Conflict snapshot of a Bopai config file, timestamped with a month marker. This is a backup/conflict artifact, not the primary config. |
| `EC_tool_sys/manager/configs/bopai/bopai_LAPTOP-MH5RB5C6_6月-03-044700-2025_Conflict.json` | file | `6月` | `June` | Another conflict snapshot of a Bopai config file, with the month marker preserved in the filename. |
| `EC_tool_sys/manager/__pycache__/__init__.cpython-311_SNOWWW_4月-18-155223-2025_Conflict.pyc` | file | `4月` | `April` | Compiled Python cache conflict artifact. Generated binary file, not source code. |

## Notes

- The three `MG400` directories appear to be robot-program/workcell folders with files such as `point.json`, `src0.lua`, and `scratch.xml`.
- The `异常序号` files are likely lab/test bookkeeping files rather than runtime code assets.
- The `4月`, `5月`, and `6月` filename fragments are month markers embedded in conflict or generated artifact filenames.

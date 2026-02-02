# Fix for React Native Bundling Error

## Problem
When running the mobile app with `npm start` or building for Android, the following error occurred:

```
Android Bundling failed 7284ms index.ts (874 modules)
Unable to resolve "./unsupportedIterableToArray.js" from "node_modules/@babel/runtime/helpers/slicedToArray.js"
  1 | var arrayWithHoles = require("./arrayWithHoles.js");
  2 | var iterableToArrayLimit = require("./iterableToArrayLimit.js");
> 3 | var unsupportedIterableToArray = require("./unsupportedIterableToArray.js");
    |                                           ^
  4 | var nonIterableRest = require("./nonIterableRest.js");
```

## Root Cause

The `@babel/runtime` package was not explicitly declared as a dependency in the mobile app's `package.json`. This package contains essential helper functions used by Babel transpilation, which is required by:
- Expo framework
- React Native 
- Many other dependencies

While some packages have `@babel/runtime` as a peer dependency, it wasn't being installed because it wasn't listed in the main dependencies.

## Solution

### 1. Added @babel/runtime to package.json

```json
"dependencies": {
  "@babel/runtime": "^7.26.0",
  // ... other dependencies
}
```

### 2. Clean Reinstall

Removed old node_modules and reinstalled:
```bash
rm -rf node_modules package-lock.json
npm install
```

### 3. Verification

After installation:
- ✅ `@babel/runtime@7.28.6` installed successfully
- ✅ `unsupportedIterableToArray.js` file now exists
- ✅ `slicedToArray.js` can resolve all dependencies
- ✅ No bundling errors

## Why This Happened

React Native and Expo apps use Babel to transpile modern JavaScript/TypeScript code. The transpilation process generates helper code that references functions from `@babel/runtime`. Without this package explicitly installed, the bundler cannot resolve these helper modules.

This is a common issue when:
1. Creating new Expo projects with certain configurations
2. Upgrading dependencies
3. Using TypeScript with strict module resolution

## Prevention

The fix is now permanent as `@babel/runtime` is listed in the dependencies. Future installations will automatically include it.

## Testing

To verify the fix works:

```bash
cd cardamom-mobile-app
npm start
```

The Metro bundler should start successfully without module resolution errors.

## Additional Troubleshooting

If you still encounter bundling issues:

1. **Clear Metro cache**:
   ```bash
   npm start -- --clear
   # or
   expo start -c
   ```

2. **Clear watchman cache** (if using watchman):
   ```bash
   watchman watch-del-all
   ```

3. **Reset Metro bundler**:
   ```bash
   rm -rf /tmp/metro-*
   npm start
   ```

4. **Complete clean install**:
   ```bash
   rm -rf node_modules package-lock.json
   npm install
   npm start
   ```

## References

- [Babel Runtime Documentation](https://babeljs.io/docs/en/babel-runtime)
- [Expo Troubleshooting](https://docs.expo.dev/troubleshooting/clear-cache-windows/)
- [React Native Metro Bundler](https://facebook.github.io/metro/)
